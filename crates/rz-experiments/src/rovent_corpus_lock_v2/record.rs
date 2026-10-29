//! Separate corpus declarations and records for the original /2 input lock.
//! These wire values are evidence metadata; parsing or validation cannot
//! recreate a live root, a native acceptance permit, or teacher authority.

use crate::pals_data_v3::{GENERATION_MAX_ROW_BYTES, generation_digest};
use crate::rovent_corpus::{
    INDEX_DOMAIN, MAX_CORPUS_ROWS, RoventCorpusChunk, RoventCorpusIndex, RoventCorpusOrigin,
    RoventTeacherPin,
};
use crate::rovent_corpus_v2::{RoventCorpusManifestV2, RoventTrainingRecordV2};
use crate::{
    ManifestError, RoventExternalProfilePin, RoventExternalPurpose, RoventInputLockV2,
    RoventPurpose, rovent_source_aggregate_sha256,
};
use serde::{Deserialize, Serialize};

pub const ROVENT_CORPUS_MANIFEST_LOCK_V2_DOMAIN: &str = "rz-rovent-corpus-manifest-lock-v2/1";
pub const ROVENT_TRAINING_RECORD_LOCK_V2_DOMAIN: &str = "rz-rovent-training-record-lock-v2/1";
pub const ROVENT_CORPUS_INDEX_LOCK_V2_DOMAIN: &str = "rz-rovent-corpus-index-lock-v2/1";
/// Independent of the row-count bound: manifest and index must also satisfy
/// this serialized metadata limit.
pub const MAX_LOCK_V2_METADATA_BYTES: usize = 16 * 1024 * 1024;

fn require(condition: bool, detail: &'static str) -> Result<(), ManifestError> {
    if condition {
        Ok(())
    } else {
        Err(ManifestError::Integrity(detail.into()))
    }
}

// Registration is a declaration, not a live teacher observation. Still, a
// teacher named by the corpus must be one of the frozen profiles in the same
// execution lock. Options/assets need the independently verified profile.
fn teacher_is_locked(teacher: &RoventTeacherPin, roster: &[RoventExternalProfilePin]) -> bool {
    roster.iter().any(|pin| {
        pin.purpose == RoventExternalPurpose::Teacher
            && pin.source_commit == teacher.source_commit
            && pin.binary.sha256 == teacher.binary_sha256
            && pin.binary.bytes == teacher.binary_bytes
            && pin.profile.sha256 == teacher.profile_sha256
            && pin.canonical_sha256 == teacher.canonical_sha256
            && pin.capability_sha256 == teacher.capability_sha256
    })
}

/// Registration data and the exact /2 lock semantic seal, kept in a new
/// domain. This is not the original bytes of the lock file nor proof that an
/// actual process used it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventCorpusManifestLockV2 {
    pub schema: String,
    pub registration: RoventCorpusManifestV2,
    pub startup_lock_sha256: String,
}

impl RoventCorpusManifestLockV2 {
    /// Decode with a unique-key/domain check. This does not establish its
    /// association with a lock; callers must subsequently call `validate`.
    pub fn from_json(input: &str) -> Result<Self, ManifestError> {
        let value: Self = crate::parse_bounded(input, MAX_LOCK_V2_METADATA_BYTES)?;
        require(
            value.schema == ROVENT_CORPUS_MANIFEST_LOCK_V2_DOMAIN,
            "explicit /2 lock corpus domain required",
        )?;
        value.registration.validate()?;
        Ok(value)
    }

    /// Check source declarations against the explicitly supplied /2 lock;
    /// actual loaded runtime and original file custody require other evidence.
    pub fn validate(&self, lock: &RoventInputLockV2) -> Result<(), ManifestError> {
        lock.validate()?;
        self.registration.validate()?;
        let engine = &lock.manifest.engine;
        let source =
            rovent_source_aggregate_sha256(&engine.source_commit, &engine.source_components)?;
        require(
            self.schema == ROVENT_CORPUS_MANIFEST_LOCK_V2_DOMAIN
                && lock.manifest.purpose == RoventPurpose::Collection
                && self.startup_lock_sha256 == lock.canonical_sha256
                && self.registration.source_commit == engine.source_commit
                && self.registration.source_sha256 == source,
            "corpus /2 lock and source declarations differ",
        )?;
        require(
            self.registration
                .teachers
                .values()
                .all(|teacher| teacher_is_locked(teacher, &engine.teachers)),
            "corpus teacher is not registered in the original /2 lock",
        )?;
        require(
            serde_json::to_vec(self)
                .map_err(|error| ManifestError::Integrity(error.to_string()))?
                .len()
                <= MAX_LOCK_V2_METADATA_BYTES,
            "corpus /2 lock manifest serialized byte cap",
        )
    }

    pub fn sha256(&self, lock: &RoventInputLockV2) -> Result<String, ManifestError> {
        self.validate(lock)?;
        generation_digest(ROVENT_CORPUS_MANIFEST_LOCK_V2_DOMAIN, self)
    }
}

#[cfg(test)]
mod teacher_lock_tests {
    use super::*;
    use crate::{ArtifactRef, ROVENT_EXTERNAL_PROFILE_DOMAIN};
    use std::collections::BTreeMap;

    fn artifact(sha: char, bytes: u64) -> ArtifactRef {
        ArtifactRef {
            path: "fixture/file".into(),
            sha256: sha.to_string().repeat(64),
            bytes,
            source: "https://example.org/fixture".into(),
            license: "MIT".into(),
        }
    }

    #[test]
    fn teacher_must_match_the_original_lock_profile_and_binary() {
        let teacher = RoventTeacherPin {
            engine: "stockfish".into(),
            source_commit: "a".repeat(40),
            binary_sha256: "b".repeat(64),
            binary_bytes: 123,
            profile_sha256: "c".repeat(64),
            canonical_sha256: "d".repeat(64),
            capability_sha256: "e".repeat(64),
            options: BTreeMap::new(),
            assets: BTreeMap::new(),
        };
        let pin = RoventExternalProfilePin {
            domain: ROVENT_EXTERNAL_PROFILE_DOMAIN.into(),
            profile: artifact('c', 12),
            canonical_sha256: teacher.canonical_sha256.clone(),
            capability_sha256: teacher.capability_sha256.clone(),
            purpose: RoventExternalPurpose::Teacher,
            binary: artifact('b', 123),
            source_commit: teacher.source_commit.clone(),
            requested_threads: 1,
            requested_hash_mib: 256,
        };
        assert!(teacher_is_locked(&teacher, std::slice::from_ref(&pin)));
        assert!(!teacher_is_locked(&teacher, &[]));
        for changed in [
            {
                let mut value = pin.clone();
                value.purpose = RoventExternalPurpose::ResearchChecker;
                value
            },
            {
                let mut value = pin.clone();
                value.binary.bytes += 1;
                value
            },
            {
                let mut value = pin.clone();
                value.binary.sha256 = "f".repeat(64);
                value
            },
            {
                let mut value = pin.clone();
                value.profile.sha256 = "f".repeat(64);
                value
            },
            {
                let mut value = pin.clone();
                value.source_commit = "f".repeat(40);
                value
            },
            {
                let mut value = pin.clone();
                value.canonical_sha256 = "f".repeat(64);
                value
            },
            {
                let mut value = pin.clone();
                value.capability_sha256 = "f".repeat(64);
                value
            },
        ] {
            assert!(!teacher_is_locked(&teacher, &[changed]));
        }
    }
}

/// The payload keeps the existing submitted-input and head data types, while
/// the required record schema and hash domain are new. It is never converted
/// into or published as a record/2. Native/root validation is distinct.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RoventTrainingRecordLockV2(pub RoventTrainingRecordV2);

impl RoventTrainingRecordLockV2 {
    /// Decode with a unique-key/domain check. Full native, target and original
    /// lock validation is a separate mandatory `validate` call.
    pub fn from_json(input: &str) -> Result<Self, ManifestError> {
        require(
            !input.is_empty() && input.len() <= GENERATION_MAX_ROW_BYTES,
            "bounded /2 lock record JSON required",
        )?;
        let value: Self = crate::decode_json(input)?;
        require(
            value.0.schema == ROVENT_TRAINING_RECORD_LOCK_V2_DOMAIN,
            "explicit /2 lock record domain required",
        )?;
        Ok(value)
    }

    pub fn sha256(&self) -> Result<String, ManifestError> {
        self.0
            .sha256_for_domain(ROVENT_TRAINING_RECORD_LOCK_V2_DOMAIN)
    }

    pub fn validate(
        &self,
        manifest: &RoventCorpusManifestLockV2,
        lock: &RoventInputLockV2,
    ) -> Result<(), ManifestError> {
        manifest.validate(lock)?;
        self.0.input.validate()?;
        super::validate_native_lifecycle_lock_v2(&self.0.native, &manifest.registration, lock)?;
        self.0.validate_record_body(
            &manifest.registration,
            ROVENT_TRAINING_RECORD_LOCK_V2_DOMAIN,
        )
    }

    /// Prepare only the new record's seals; the native evidence must already
    /// be present and pass its own /2 validator. This issues no authority.
    pub fn seal(
        &mut self,
        manifest: &RoventCorpusManifestLockV2,
        lock: &RoventInputLockV2,
    ) -> Result<(), ManifestError> {
        require(
            self.0.schema == ROVENT_TRAINING_RECORD_LOCK_V2_DOMAIN,
            "explicit /2 lock record domain required",
        )?;
        self.0.input_sha256 = self.0.input.sha256()?;
        self.0.origin_sha256 = self.0.origin(&manifest.registration)?;
        self.0.record_sha256 = self.sha256()?;
        self.validate(manifest, lock)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventCorpusIndexLockV2 {
    pub schema: String,
    pub manifest_sha256: String,
    pub record_count: u64,
    pub chunks: Vec<RoventCorpusChunk>,
    pub origins: Vec<RoventCorpusOrigin>,
}

impl RoventCorpusIndexLockV2 {
    /// Decode with a unique-key/domain check. Counts and the original
    /// manifest/lock join are checked only by the separate `validate` call.
    pub fn from_json(input: &str) -> Result<Self, ManifestError> {
        let value: Self = crate::parse_bounded(input, MAX_LOCK_V2_METADATA_BYTES)?;
        require(
            value.schema == ROVENT_CORPUS_INDEX_LOCK_V2_DOMAIN,
            "explicit /2 lock index domain required",
        )?;
        Ok(value)
    }

    pub fn validate(
        &self,
        manifest: &RoventCorpusManifestLockV2,
        lock: &RoventInputLockV2,
    ) -> Result<(), ManifestError> {
        require(
            self.schema == ROVENT_CORPUS_INDEX_LOCK_V2_DOMAIN
                && self.manifest_sha256 == manifest.sha256(lock)?
                && self.record_count <= MAX_CORPUS_ROWS as u64
                && self.origins.len() as u64 == self.record_count
                && self
                    .chunks
                    .iter()
                    .try_fold(0u64, |sum, chunk| sum.checked_add(chunk.records))
                    == Some(self.record_count),
            "corpus /2 lock index domain/manifest/count differs",
        )?;
        require(
            serde_json::to_vec(self)
                .map_err(|error| ManifestError::Integrity(error.to_string()))?
                .len()
                <= MAX_LOCK_V2_METADATA_BYTES,
            "corpus /2 lock index serialized byte cap",
        )?;
        // Reuse only the existing bounds/uniqueness calculation. This value is
        // never returned, hashed as /1, or published as an older index.
        RoventCorpusIndex {
            schema: INDEX_DOMAIN.into(),
            manifest_sha256: self.manifest_sha256.clone(),
            record_count: self.record_count,
            chunks: self.chunks.clone(),
            origins: self.origins.clone(),
        }
        .validate()
    }

    pub fn sha256(&self) -> Result<String, ManifestError> {
        generation_digest(ROVENT_CORPUS_INDEX_LOCK_V2_DOMAIN, self)
    }
}
