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
pub const ROVENT_CORPUS_MANIFEST_CPU_LOCK_V3_DOMAIN: &str =
    "rz-rovent-corpus-manifest-cpu-lock-v3/1";
pub const ROVENT_TRAINING_RECORD_CPU_LOCK_V3_DOMAIN: &str =
    "rz-rovent-training-record-cpu-lock-v3/1";
pub const ROVENT_CORPUS_INDEX_CPU_LOCK_V3_DOMAIN: &str = "rz-rovent-corpus-index-cpu-lock-v3/1";
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
            value.schema == ROVENT_CORPUS_MANIFEST_LOCK_V3_DOMAIN
                || value.schema == ROVENT_CORPUS_MANIFEST_CPU_LOCK_V3_DOMAIN,
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
        // The current corpus registration format requires an exact source
        // declaration. A functional /3 run may legitimately have no commit;
        // do not fabricate one or interpret this as a product execution error.
        let source_commit = engine.source_commit.as_deref().ok_or_else(|| {
            ManifestError::Integrity("corpus source identity is unavailable".into())
        })?;
        let source = rovent_source_aggregate_sha256(source_commit, &engine.source_components)?;
        let (expected_schema, runtime_sha) = match &lock.manifest.system_runtime_lock.runtime {
            NativeRuntimeSourceV6::SystemCuda(runtime) => {
                let core = runtime
                    .libraries
                    .iter()
                    .find(|library| library.role == "ort-core")
                    .ok_or_else(|| {
                        ManifestError::Integrity("system ORT core registration missing".into())
                    })?;
                (ROVENT_CORPUS_MANIFEST_LOCK_V3_DOMAIN, core.sha256.as_str())
            }
            NativeRuntimeSourceV6::Cpu => {
                let pc = &lock.manifest.engine.pc;
                require(
                    pc.backend == crate::RoventBackend::OrtCpu
                        && pc.device == crate::RoventDevice::Cpu
                        && pc.runtime_artifacts.len() == 1,
                    "CPU collection requires the single pinned CPU ORT runtime",
                )?;
                (
                    ROVENT_CORPUS_MANIFEST_CPU_LOCK_V3_DOMAIN,
                    pc.runtime_artifacts[0].sha256.as_str(),
                )
            }
        };
        require(
            self.schema == expected_schema
                && lock.manifest.purpose == RoventPurpose::Collection
                && self.startup_lock_sha256 == lock.canonical_sha256
                && self.registration.source_commit == source_commit
                && self.registration.source_sha256 == source
                && self.registration.runtime_sha256 == runtime_sha,
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
        generation_digest(&self.schema, self)
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
            value.0.schema == ROVENT_TRAINING_RECORD_LOCK_V3_DOMAIN
                || value.0.schema == ROVENT_TRAINING_RECORD_CPU_LOCK_V3_DOMAIN,
            "explicit /3 lock record domain required",
        )?;
        Ok(value)
    }
    pub fn sha256(&self) -> Result<String, ManifestError> {
        require(
            self.0.schema == ROVENT_TRAINING_RECORD_LOCK_V3_DOMAIN
                || self.0.schema == ROVENT_TRAINING_RECORD_CPU_LOCK_V3_DOMAIN,
            "explicit /3 lock record domain required",
        )?;
        self.0.sha256_for_domain(&self.0.schema)
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
        let expected = if matches!(
            lock.manifest.system_runtime_lock.runtime,
            NativeRuntimeSourceV6::Cpu
        ) {
            ROVENT_TRAINING_RECORD_CPU_LOCK_V3_DOMAIN
        } else {
            ROVENT_TRAINING_RECORD_LOCK_V3_DOMAIN
        };
        require(self.0.schema == expected, "record runtime family differs")?;
        self.0.input.validate()?;
        super::validate_native_lifecycle_lock_v3(&self.0.native, &manifest.registration, lock)?;
        self.0
            .validate_record_body(&manifest.registration, expected)
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
            self.0.schema
                == if matches!(
                    lock.manifest.system_runtime_lock.runtime,
                    NativeRuntimeSourceV6::Cpu
                ) {
                    ROVENT_TRAINING_RECORD_CPU_LOCK_V3_DOMAIN
                } else {
                    ROVENT_TRAINING_RECORD_LOCK_V3_DOMAIN
                },
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
            value.schema == ROVENT_CORPUS_INDEX_LOCK_V3_DOMAIN
                || value.schema == ROVENT_CORPUS_INDEX_CPU_LOCK_V3_DOMAIN,
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
            self.schema
                == if matches!(
                    lock.manifest.system_runtime_lock.runtime,
                    NativeRuntimeSourceV6::Cpu
                ) {
                    ROVENT_CORPUS_INDEX_CPU_LOCK_V3_DOMAIN
                } else {
                    ROVENT_CORPUS_INDEX_LOCK_V3_DOMAIN
                }
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
        require(
            self.schema == ROVENT_CORPUS_INDEX_LOCK_V3_DOMAIN
                || self.schema == ROVENT_CORPUS_INDEX_CPU_LOCK_V3_DOMAIN,
            "explicit /3 lock index domain required",
        )?;
        generation_digest(&self.schema, self)
    }
}

#[cfg(test)]
mod cpu_family_tests {
    use super::*;

    #[test]
    fn index_digest_rejects_unregistered_family_and_separates_cpu_cuda() {
        let mut index = RoventCorpusIndexLockV3 {
            schema: ROVENT_CORPUS_INDEX_LOCK_V3_DOMAIN.into(),
            manifest_sha256: "a".repeat(64),
            record_count: 0,
            chunks: vec![],
            origins: vec![],
        };
        let cuda = index.sha256().unwrap();
        index.schema = ROVENT_CORPUS_INDEX_CPU_LOCK_V3_DOMAIN.into();
        let cpu = index.sha256().unwrap();
        assert_ne!(cuda, cpu);
        index.schema = "unregistered-domain".into();
        assert!(index.sha256().is_err());
    }
}
