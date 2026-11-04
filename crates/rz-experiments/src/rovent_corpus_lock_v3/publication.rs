//! /3 publication metadata. Validating it neither reads publication bytes
//! nor recreates a native owner, physical fence, or source provenance.
use crate::pals_data_v3::generation_sha;
use crate::rovent_corpus_lock_v2::publication::{
    ROVENT_LOCK_V2_PUBLICATION_DOMAIN, RoventLockV2Publication, RoventLockV2PublicationAdmission,
};
use crate::{MAX_MANIFEST_BYTES, ManifestError, decode_json};
use serde::{Deserialize, Serialize};

pub use crate::rovent_corpus_lock_v2::publication::{
    RoventLockV2InputLockBytesOrigin as RoventLockV3InputLockBytesOrigin,
    RoventLockV2PublicationFile as RoventLockV3PublicationFile,
};

pub const ROVENT_LOCK_V3_PUBLICATION_DOMAIN: &str = "rz-rovent-corpus-publication-lock-v3/1";

fn require(condition: bool, detail: &'static str) -> Result<(), ManifestError> {
    if condition {
        Ok(())
    } else {
        Err(ManifestError::Integrity(detail.into()))
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventLockV3PublicationAdmission {
    pub native_admission_sha256: String,
    pub input_lock: RoventLockV3PublicationFile,
    pub input_lock_bytes_origin: RoventLockV3InputLockBytesOrigin,
    pub canonical_input_lock_sha256: String,
    /// Captured system identity declared by this admission. It still requires
    /// independent verification against the original /3 lock and child.
    pub system_runtime_identity_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventLockV3Publication {
    pub schema: String,
    pub manifest: RoventLockV3PublicationFile,
    pub index: RoventLockV3PublicationFile,
    pub chunks: Vec<RoventLockV3PublicationFile>,
    pub canonical_manifest_sha256: String,
    pub canonical_index_sha256: String,
    pub source_commit: String,
    pub source_sha256: String,
    pub runtime_sha256: String,
    pub record_count: u64,
    pub total_payload_bytes: u64,
    pub native_admission_sha256: Vec<String>,
    pub admissions: Vec<RoventLockV3PublicationAdmission>,
}

impl RoventLockV3Publication {
    pub fn from_json(input: &str) -> Result<Self, ManifestError> {
        let value: Self = decode_json(input)?;
        value.validate()?;
        Ok(value)
    }

    /// Structural/path/budget validation only. The real lock, system identity,
    /// child observations, corpus rows and bytes are verified independently.
    pub fn validate(&self) -> Result<(), ManifestError> {
        require(
            self.schema == ROVENT_LOCK_V3_PUBLICATION_DOMAIN
                && self
                    .admissions
                    .iter()
                    .all(|a| generation_sha(&a.system_runtime_identity_sha256)),
            "lock-v3 publication domain or system runtime identity",
        )?;
        // Bound the actual /3 envelope before allocating the shared projection.
        struct Count(usize);
        impl std::io::Write for Count {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.0 = self
                    .0
                    .checked_add(bytes.len())
                    .filter(|n| *n <= MAX_MANIFEST_BYTES)
                    .ok_or_else(|| std::io::Error::other("publication JSON byte cap"))?;
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        serde_json::to_writer(Count(0), self)
            .map_err(|e| ManifestError::Integrity(e.to_string()))?;
        // Reuse the established path/byte/uniqueness checks in a private
        // temporary projection. It is never serialized or published as /2.
        let common = RoventLockV2Publication {
            schema: ROVENT_LOCK_V2_PUBLICATION_DOMAIN.into(),
            manifest: self.manifest.clone(),
            index: self.index.clone(),
            chunks: self.chunks.clone(),
            canonical_manifest_sha256: self.canonical_manifest_sha256.clone(),
            canonical_index_sha256: self.canonical_index_sha256.clone(),
            source_commit: self.source_commit.clone(),
            source_sha256: self.source_sha256.clone(),
            runtime_sha256: self.runtime_sha256.clone(),
            record_count: self.record_count,
            total_payload_bytes: self.total_payload_bytes,
            native_admission_sha256: self.native_admission_sha256.clone(),
            admissions: self
                .admissions
                .iter()
                .map(|a| RoventLockV2PublicationAdmission {
                    native_admission_sha256: a.native_admission_sha256.clone(),
                    input_lock: a.input_lock.clone(),
                    input_lock_bytes_origin: a.input_lock_bytes_origin,
                    canonical_input_lock_sha256: a.canonical_input_lock_sha256.clone(),
                    runtime_bundle_manifest: None,
                })
                .collect(),
        };
        common.validate()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(path: &str, symbol: char, bytes: u64) -> RoventLockV3PublicationFile {
        RoventLockV3PublicationFile {
            path: path.into(),
            sha256: symbol.to_string().repeat(64),
            bytes,
        }
    }

    #[test]
    fn publication_v3_requires_separate_domain_and_runtime_identity() {
        let mut publication = RoventLockV3Publication {
            schema: ROVENT_LOCK_V3_PUBLICATION_DOMAIN.into(),
            manifest: file("corpus-manifest.json", 'a', 10),
            index: file("corpus-index.json", 'b', 20),
            chunks: vec![file("chunks/0.jsonl", 'c', 30)],
            canonical_manifest_sha256: "a".repeat(64),
            canonical_index_sha256: "b".repeat(64),
            source_commit: "f".repeat(40),
            source_sha256: "e".repeat(64),
            runtime_sha256: "d".repeat(64),
            record_count: 1,
            total_payload_bytes: 100,
            native_admission_sha256: vec!["a".repeat(64)],
            admissions: vec![RoventLockV3PublicationAdmission {
                native_admission_sha256: "a".repeat(64),
                input_lock: file("input-locks/lock.json", 'd', 40),
                input_lock_bytes_origin: RoventLockV3InputLockBytesOrigin::OriginalFile,
                canonical_input_lock_sha256: "e".repeat(64),
                system_runtime_identity_sha256: "f".repeat(64),
            }],
        };
        let encoded = serde_json::to_string(&publication).unwrap();
        assert_eq!(
            RoventLockV3Publication::from_json(&encoded).unwrap(),
            publication
        );
        assert!(RoventLockV2Publication::from_json(&encoded).is_err());
        publication.admissions[0]
            .system_runtime_identity_sha256
            .clear();
        assert!(publication.validate().is_err());
        publication.admissions[0].system_runtime_identity_sha256 = "f".repeat(64);
        publication.schema = ROVENT_LOCK_V2_PUBLICATION_DOMAIN.into();
        assert!(publication.validate().is_err());
    }
}
