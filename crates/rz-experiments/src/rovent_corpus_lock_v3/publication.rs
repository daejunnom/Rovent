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
pub const ROVENT_LOCK_V3_CPU_PUBLICATION_DOMAIN: &str =
    "rz-rovent-corpus-publication-cpu-lock-v3/1";

fn require(condition: bool, detail: &'static str) -> Result<(), ManifestError> {
    if condition {
        Ok(())
    } else {
        Err(ManifestError::Integrity(detail.into()))
    }
}

// An absent family-specific field is allowed, but an explicitly present
// value must remain a string. In particular, accepting null would silently
// erase a wire field during reserialization and widen the historical parser.
fn present_string<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Option<String>, D::Error> {
    String::deserialize(d).map(Some)
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
    #[serde(
        default,
        deserialize_with = "present_string",
        skip_serializing_if = "Option::is_none"
    )]
    pub system_runtime_identity_sha256: Option<String>,
    /// The single actual CPU ORT library SHA, not a CUDA system identity.
    #[serde(
        default,
        deserialize_with = "present_string",
        skip_serializing_if = "Option::is_none"
    )]
    pub cpu_runtime_library_sha256: Option<String>,
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
            match self.schema.as_str() {
                ROVENT_LOCK_V3_PUBLICATION_DOMAIN => self.admissions.iter().all(|a| {
                    a.system_runtime_identity_sha256
                        .as_ref()
                        .is_some_and(|s| generation_sha(s))
                        && a.cpu_runtime_library_sha256.is_none()
                }),
                ROVENT_LOCK_V3_CPU_PUBLICATION_DOMAIN => self.admissions.iter().all(|a| {
                    a.system_runtime_identity_sha256.is_none()
                        && a.cpu_runtime_library_sha256
                            .as_ref()
                            .is_some_and(|s| generation_sha(s) && s == &self.runtime_sha256)
                }),
                _ => false,
            },
            "lock-v3 publication domain or runtime identity",
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
                system_runtime_identity_sha256: Some("f".repeat(64)),
                cpu_runtime_library_sha256: None,
            }],
        };
        let encoded = serde_json::to_string(&publication).unwrap();
        // The legacy CUDA wire had exactly one required string identity field.
        // Compare byte-for-byte against that independently declared shape.
        #[derive(Serialize)]
        struct LegacyAdmission<'a> {
            native_admission_sha256: &'a str,
            input_lock: &'a RoventLockV3PublicationFile,
            input_lock_bytes_origin: RoventLockV3InputLockBytesOrigin,
            canonical_input_lock_sha256: &'a str,
            system_runtime_identity_sha256: &'a str,
        }
        #[derive(Serialize)]
        struct LegacyPublication<'a> {
            schema: &'a str,
            manifest: &'a RoventLockV3PublicationFile,
            index: &'a RoventLockV3PublicationFile,
            chunks: &'a [RoventLockV3PublicationFile],
            canonical_manifest_sha256: &'a str,
            canonical_index_sha256: &'a str,
            source_commit: &'a str,
            source_sha256: &'a str,
            runtime_sha256: &'a str,
            record_count: u64,
            total_payload_bytes: u64,
            native_admission_sha256: &'a [String],
            admissions: Vec<LegacyAdmission<'a>>,
        }
        let a = &publication.admissions[0];
        let legacy = LegacyPublication {
            schema: &publication.schema,
            manifest: &publication.manifest,
            index: &publication.index,
            chunks: &publication.chunks,
            canonical_manifest_sha256: &publication.canonical_manifest_sha256,
            canonical_index_sha256: &publication.canonical_index_sha256,
            source_commit: &publication.source_commit,
            source_sha256: &publication.source_sha256,
            runtime_sha256: &publication.runtime_sha256,
            record_count: publication.record_count,
            total_payload_bytes: publication.total_payload_bytes,
            native_admission_sha256: &publication.native_admission_sha256,
            admissions: vec![LegacyAdmission {
                native_admission_sha256: &a.native_admission_sha256,
                input_lock: &a.input_lock,
                input_lock_bytes_origin: a.input_lock_bytes_origin,
                canonical_input_lock_sha256: &a.canonical_input_lock_sha256,
                system_runtime_identity_sha256: a
                    .system_runtime_identity_sha256
                    .as_deref()
                    .unwrap(),
            }],
        };
        assert_eq!(encoded, serde_json::to_string(&legacy).unwrap());
        assert!(!encoded.contains("cpu_runtime_library_sha256"));
        assert_eq!(
            RoventLockV3Publication::from_json(&encoded).unwrap(),
            publication
        );
        assert!(RoventLockV2Publication::from_json(&encoded).is_err());
        let mut cuda_null: serde_json::Value = serde_json::from_str(&encoded).unwrap();
        cuda_null["admissions"][0]["cpu_runtime_library_sha256"] = serde_json::Value::Null;
        assert!(RoventLockV3Publication::from_json(&cuda_null.to_string()).is_err());
        cuda_null["admissions"][0]["system_runtime_identity_sha256"] = serde_json::Value::Null;
        assert!(RoventLockV3Publication::from_json(&cuda_null.to_string()).is_err());
        publication.admissions[0].system_runtime_identity_sha256 = Some(String::new());
        assert!(publication.validate().is_err());
        publication.admissions[0].system_runtime_identity_sha256 = Some("f".repeat(64));
        publication.schema = ROVENT_LOCK_V3_CPU_PUBLICATION_DOMAIN.into();
        assert!(publication.validate().is_err());
        publication.admissions[0].system_runtime_identity_sha256 = None;
        publication.admissions[0].cpu_runtime_library_sha256 =
            Some(publication.runtime_sha256.clone());
        assert!(publication.validate().is_ok());
        let cpu = serde_json::to_string(&publication).unwrap();
        assert!(!cpu.contains("system_runtime_identity_sha256"));
        let mut cpu_null: serde_json::Value = serde_json::from_str(&cpu).unwrap();
        cpu_null["admissions"][0]["system_runtime_identity_sha256"] = serde_json::Value::Null;
        assert!(RoventLockV3Publication::from_json(&cpu_null.to_string()).is_err());
        let mut cpu_required_null: serde_json::Value = serde_json::from_str(&cpu).unwrap();
        cpu_required_null["admissions"][0]["cpu_runtime_library_sha256"] = serde_json::Value::Null;
        assert!(RoventLockV3Publication::from_json(&cpu_required_null.to_string()).is_err());
        assert_eq!(
            RoventLockV3Publication::from_json(&cpu).unwrap(),
            publication
        );
        publication.admissions[0].cpu_runtime_library_sha256 = Some("0".repeat(64));
        assert!(publication.validate().is_err());
        publication.schema = ROVENT_LOCK_V2_PUBLICATION_DOMAIN.into();
        assert!(publication.validate().is_err());
    }
}
