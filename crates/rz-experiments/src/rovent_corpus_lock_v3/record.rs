//! P/C corpus records scoped to the original system-runtime /3 input lock.
//! V observations are not converted into P/C rows or task targets.
use crate::pals_data_v3::{GENERATION_MAX_ROW_BYTES, GenerationRoleV3, generation_digest};
use crate::rovent_corpus::{
    INDEX_DOMAIN, MAX_CORPUS_ROWS, RoventCorpusChunk, RoventCorpusIndex, RoventCorpusOrigin,
};
use crate::rovent_corpus_lock_v2::record::teacher_is_locked;
use crate::rovent_corpus_v2::{RoventCorpusManifestV2, RoventTrainingRecordV2};
use crate::{
    ManifestError, NativeRuntimeSourceV6, RoventInputLockV3, RoventPurpose,
    rovent_source_aggregate_sha256,
};
use serde::{Deserialize, Serialize};

pub const ROVENT_CORPUS_MANIFEST_LOCK_V3_DOMAIN: &str = "rz-rovent-corpus-manifest-lock-v3/1";
pub const ROVENT_TRAINING_RECORD_LOCK_V3_DOMAIN: &str = "rz-rovent-training-record-lock-v3/1";
pub const ROVENT_CORPUS_INDEX_LOCK_V3_DOMAIN: &str = "rz-rovent-corpus-index-lock-v3/1";
pub const MAX_LOCK_V3_METADATA_BYTES: usize =
    crate::rovent_corpus_lock_v2::record::MAX_LOCK_V2_METADATA_BYTES;

fn require(condition: bool, detail: &'static str) -> Result<(), ManifestError> {
    if condition {
        Ok(())
    } else {
        Err(ManifestError::Integrity(detail.into()))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventCorpusManifestLockV3 {
    pub schema: String,
    pub registration: RoventCorpusManifestV2,
    pub startup_lock_sha256: String,
}

impl RoventCorpusManifestLockV3 {
    pub fn from_json(input: &str) -> Result<Self, ManifestError> {
        let value: Self = crate::parse_bounded(input, MAX_LOCK_V3_METADATA_BYTES)?;
        require(
            value.schema == ROVENT_CORPUS_MANIFEST_LOCK_V3_DOMAIN,
            "explicit /3 lock corpus domain required",
        )?;
        value.registration.validate()?;
        Ok(value)
    }

    /// Source is the lock's declared source identity, not proof of the actual
    /// compiled binary or eligibility for training.
    pub fn validate(&self, lock: &RoventInputLockV3) -> Result<(), ManifestError> {
        lock.validate()?;
        self.registration.validate()?;
        let engine = &lock.manifest.engine;
        let source =
            rovent_source_aggregate_sha256(&engine.source_commit, &engine.source_components)?;
        let runtime = match &lock.manifest.system_runtime_lock.runtime {
            NativeRuntimeSourceV6::SystemCuda(runtime) => runtime,
            NativeRuntimeSourceV6::Cpu => {
                return Err(ManifestError::Integrity(
                    "system CUDA /3 collection requires an explicit system CUDA runtime".into(),
                ));
            }
        };
        let core = runtime
            .libraries
            .iter()
            .find(|library| library.role == "ort-core")
            .ok_or_else(|| {
                ManifestError::Integrity("system ORT core registration missing".into())
            })?;
        require(
            self.schema == ROVENT_CORPUS_MANIFEST_LOCK_V3_DOMAIN
                && lock.manifest.purpose == RoventPurpose::Collection
                && self.startup_lock_sha256 == lock.canonical_sha256
                && self.registration.source_commit == engine.source_commit
                && self.registration.source_sha256 == source
                && self.registration.runtime_sha256 == core.sha256,
            "corpus /3 lock, declared source or runtime registration differs",
        )?;
        require(
            self.registration
                .teachers
                .values()
                .all(|teacher| teacher_is_locked(teacher, &engine.teachers)),
            "corpus teacher is not registered in the original /3 lock",
        )?;
        require(
            serde_json::to_vec(self)
                .map_err(|error| ManifestError::Integrity(error.to_string()))?
                .len()
                <= MAX_LOCK_V3_METADATA_BYTES,
            "corpus /3 lock manifest serialized byte cap",
        )
    }

    pub fn sha256(&self, lock: &RoventInputLockV3) -> Result<String, ManifestError> {
        self.validate(lock)?;
        generation_digest(ROVENT_CORPUS_MANIFEST_LOCK_V3_DOMAIN, self)
    }
}

/// Reuses the P/C payload without relabelling it as an older record family.
/// Validator input/action/outcome records require their own future domain.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RoventTrainingRecordLockV3(pub RoventTrainingRecordV2);

impl RoventTrainingRecordLockV3 {
    pub fn from_json(input: &str) -> Result<Self, ManifestError> {
        require(
            !input.is_empty() && input.len() <= GENERATION_MAX_ROW_BYTES,
            "bounded /3 lock record JSON required",
        )?;
        let value: Self = crate::decode_json(input)?;
        require(
            value.0.schema == ROVENT_TRAINING_RECORD_LOCK_V3_DOMAIN,
            "explicit /3 lock record domain required",
        )?;
        Ok(value)
    }
    pub fn sha256(&self) -> Result<String, ManifestError> {
        self.0
            .sha256_for_domain(ROVENT_TRAINING_RECORD_LOCK_V3_DOMAIN)
    }
    pub fn validate(
        &self,
        manifest: &RoventCorpusManifestLockV3,
        lock: &RoventInputLockV3,
    ) -> Result<(), ManifestError> {
        if self.0.role == GenerationRoleV3::Validator {
            return Err(ManifestError::Integrity(
                "unsupported V input/action/outcome in P/C lock-v3 corpus".into(),
            ));
        }
        manifest.validate(lock)?;
        self.0.input.validate()?;
        super::validate_native_lifecycle_lock_v3(&self.0.native, &manifest.registration, lock)?;
        self.0.validate_record_body(
            &manifest.registration,
            ROVENT_TRAINING_RECORD_LOCK_V3_DOMAIN,
        )
    }
    pub fn seal(
        &mut self,
        manifest: &RoventCorpusManifestLockV3,
        lock: &RoventInputLockV3,
    ) -> Result<(), ManifestError> {
        if self.0.role == GenerationRoleV3::Validator {
            return Err(ManifestError::Integrity(
                "unsupported V input/action/outcome in P/C lock-v3 corpus".into(),
            ));
        }
        require(
            self.0.schema == ROVENT_TRAINING_RECORD_LOCK_V3_DOMAIN,
            "explicit /3 lock record domain required",
        )?;
        self.0.input_sha256 = self.0.input.sha256()?;
        self.0.origin_sha256 = self.0.origin(&manifest.registration)?;
        self.0.record_sha256 = self.sha256()?;
        self.validate(manifest, lock)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventCorpusIndexLockV3 {
    pub schema: String,
    pub manifest_sha256: String,
    pub record_count: u64,
    pub chunks: Vec<RoventCorpusChunk>,
    pub origins: Vec<RoventCorpusOrigin>,
}
impl RoventCorpusIndexLockV3 {
    pub fn from_json(input: &str) -> Result<Self, ManifestError> {
        let value: Self = crate::parse_bounded(input, MAX_LOCK_V3_METADATA_BYTES)?;
        require(
            value.schema == ROVENT_CORPUS_INDEX_LOCK_V3_DOMAIN,
            "explicit /3 lock index domain required",
        )?;
        Ok(value)
    }
    pub fn validate(
        &self,
        manifest: &RoventCorpusManifestLockV3,
        lock: &RoventInputLockV3,
    ) -> Result<(), ManifestError> {
        require(
            self.schema == ROVENT_CORPUS_INDEX_LOCK_V3_DOMAIN
                && self.manifest_sha256 == manifest.sha256(lock)?
                && self.record_count <= MAX_CORPUS_ROWS as u64
                && self.origins.len() as u64 == self.record_count
                && self
                    .chunks
                    .iter()
                    .try_fold(0u64, |sum, chunk| sum.checked_add(chunk.records))
                    == Some(self.record_count),
            "corpus /3 lock index domain/manifest/count differs",
        )?;
        require(
            serde_json::to_vec(self)
                .map_err(|error| ManifestError::Integrity(error.to_string()))?
                .len()
                <= MAX_LOCK_V3_METADATA_BYTES,
            "corpus /3 lock index serialized byte cap",
        )?;
        // Shared bounds/uniqueness only, never serialized as an older index.
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
        generation_digest(ROVENT_CORPUS_INDEX_LOCK_V3_DOMAIN, self)
    }
}
