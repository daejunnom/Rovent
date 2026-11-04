//! Persisted publication metadata for the input-lock /2 corpus family.
//!
//! Validation here checks declarations and bounded payload accounting only.
//! It neither reads the payloads nor reconstructs any live authority, native
//! acceptance, physical completion or teacher-label eligibility. In particular,
//! `input_lock_bytes_origin` is a declaration: only an independent historical
//! consumer that verifies the original process and file evidence may establish
//! that published bytes were actually the original input-lock file.

use crate::pals_data_v3::generation_sha;
use crate::rovent_corpus::{MAX_CHUNK_BYTES, MAX_CORPUS_BYTES, MAX_CORPUS_ROWS};
use crate::{MAX_MANIFEST_BYTES, ManifestError, decode_json};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const ROVENT_LOCK_V2_PUBLICATION_DOMAIN: &str = "rz-rovent-corpus-publication-lock-v2/1";
const MANIFEST_PATH: &str = "corpus-manifest.json";
const INDEX_PATH: &str = "corpus-index.json";
const MAX_SOURCE_METADATA_BYTES: u64 = MAX_MANIFEST_BYTES as u64;
const MAX_BUNDLE_BYTES: u64 = 64 * 1024;
const MAX_CHUNKS: usize = 8192;

fn require(condition: bool, reason: &'static str) -> Result<(), ManifestError> {
    if condition {
        Ok(())
    } else {
        Err(ManifestError::Integrity(reason.into()))
    }
}

fn relative_file(path: &str) -> bool {
    !path.is_empty()
        && path.len() <= 4096
        && !path.contains('\\')
        && !path.contains(':')
        && !path.starts_with('/')
        && !path.chars().any(char::is_control)
        && path
            .split('/')
            .all(|part| !part.is_empty() && part != "." && part != "..")
}

/// Raw bytes of a file relative to the published directory, not its decoded
/// object or canonical semantic digest. Actual bytes are checked by readers.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventLockV2PublicationFile {
    pub path: String,
    pub sha256: String,
    pub bytes: u64,
}

impl RoventLockV2PublicationFile {
    pub fn validate(&self) -> Result<(), ManifestError> {
        require(
            relative_file(&self.path)
                && generation_sha(&self.sha256)
                && self.bytes > 0
                && self.bytes <= MAX_CORPUS_BYTES,
            "lock-v2 publication normalized file pin",
        )
    }
}

/// Whether the publisher declared this file as original bytes or a validated
/// reserialization. A declaration alone does not prove original-file custody.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoventLockV2InputLockBytesOrigin {
    OriginalFile,
    ValidatedSemanticSnapshot,
}

/// One recorded native origin and the exact metadata files associated with it.
/// The raw lock file SHA and canonical lock SHA are intentionally separate.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventLockV2PublicationAdmission {
    pub native_admission_sha256: String,
    pub input_lock: RoventLockV2PublicationFile,
    pub input_lock_bytes_origin: RoventLockV2InputLockBytesOrigin,
    pub canonical_input_lock_sha256: String,
    pub runtime_bundle_manifest: Option<RoventLockV2PublicationFile>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventLockV2Publication {
    pub schema: String,
    pub manifest: RoventLockV2PublicationFile,
    pub index: RoventLockV2PublicationFile,
    pub chunks: Vec<RoventLockV2PublicationFile>,
    pub canonical_manifest_sha256: String,
    pub canonical_index_sha256: String,
    pub source_commit: String,
    pub source_sha256: String,
    pub runtime_sha256: String,
    pub record_count: u64,
    pub total_payload_bytes: u64,
    pub native_admission_sha256: Vec<String>,
    pub admissions: Vec<RoventLockV2PublicationAdmission>,
}

impl RoventLockV2Publication {
    /// Strictly decode this domain without falling back to old publications.
    pub fn from_json(input: &str) -> Result<Self, ManifestError> {
        let publication: Self = decode_json(input)?;
        publication.validate()?;
        Ok(publication)
    }

    /// DTO shape and byte-budget validation only. It must be followed by
    /// independent file-byte, typed lock/root and historical row validation.
    pub fn validate(&self) -> Result<(), ManifestError> {
        self.manifest.validate()?;
        self.index.validate()?;
        require(
            self.schema == ROVENT_LOCK_V2_PUBLICATION_DOMAIN
                && self.manifest.path == MANIFEST_PATH
                && self.index.path == INDEX_PATH
                && self.manifest.bytes <= super::record::MAX_LOCK_V2_METADATA_BYTES as u64
                && self.index.bytes <= super::record::MAX_LOCK_V2_METADATA_BYTES as u64
                && self.source_commit.len() == 40
                && self
                    .source_commit
                    .bytes()
                    .all(|v| v.is_ascii_digit() || (b'a'..=b'f').contains(&v))
                && [
                    &self.canonical_manifest_sha256,
                    &self.canonical_index_sha256,
                    &self.source_sha256,
                    &self.runtime_sha256,
                ]
                .into_iter()
                .all(|v| generation_sha(v))
                && self.record_count > 0
                && self.record_count <= MAX_CORPUS_ROWS as u64
                && !self.chunks.is_empty()
                && self.chunks.len() <= MAX_CHUNKS
                && !self.native_admission_sha256.is_empty()
                && self.native_admission_sha256.len() <= MAX_CORPUS_ROWS
                && self
                    .native_admission_sha256
                    .iter()
                    .all(|v| generation_sha(v))
                && self
                    .native_admission_sha256
                    .windows(2)
                    .all(|pair| pair[0] < pair[1]),
            "lock-v2 publication domain/source/count/admission seals",
        )?;
        require(
            self.admissions.len() == self.native_admission_sha256.len()
                && self
                    .admissions
                    .iter()
                    .zip(&self.native_admission_sha256)
                    .all(|(entry, sha)| {
                        &entry.native_admission_sha256 == sha
                            && generation_sha(&entry.canonical_input_lock_sha256)
                    }),
            "lock-v2 publication exact admission coverage",
        )?;

        let mut names = BTreeSet::from([self.manifest.path.as_str(), self.index.path.as_str()]);
        let mut bytes = self
            .manifest
            .bytes
            .checked_add(self.index.bytes)
            .ok_or_else(|| {
                ManifestError::Integrity("lock-v2 publication byte count overflow".into())
            })?;
        for chunk in &self.chunks {
            chunk.validate()?;
            require(
                chunk.path.starts_with("chunks/")
                    && chunk.path.ends_with(".jsonl")
                    && chunk.bytes <= MAX_CHUNK_BYTES as u64
                    && names.insert(&chunk.path),
                "lock-v2 publication chunk path/cap/duplicate",
            )?;
            bytes = bytes.checked_add(chunk.bytes).ok_or_else(|| {
                ManifestError::Integrity("lock-v2 publication byte count overflow".into())
            })?;
        }

        // A canonical lock may have multiple original byte encodings. Dedup
        // identical payloads by their declared path and exact file pin; do not
        // claim that a reserialized lock is the source file's original bytes.
        let mut locks: BTreeMap<&str, (&RoventLockV2PublicationFile, &str)> = BTreeMap::new();
        let mut lock_bytes: BTreeMap<(&str, u64), &str> = BTreeMap::new();
        let mut bundles: BTreeMap<&str, &RoventLockV2PublicationFile> = BTreeMap::new();
        for entry in &self.admissions {
            entry.input_lock.validate()?;
            require(
                entry.input_lock.path.starts_with("input-locks/")
                    && entry.input_lock.path.ends_with(".json")
                    && entry.input_lock.bytes <= MAX_SOURCE_METADATA_BYTES,
                "lock-v2 publication input lock path/cap",
            )?;
            let canonical = entry.canonical_input_lock_sha256.as_str();
            let raw_identity = (entry.input_lock.sha256.as_str(), entry.input_lock.bytes);
            if let Some(existing) = lock_bytes.insert(raw_identity, canonical) {
                require(
                    existing == canonical,
                    "lock-v2 identical raw lock has conflicting canonical SHA",
                )?;
            }
            if let Some((existing, prior_canonical)) = locks.get(entry.input_lock.path.as_str()) {
                require(
                    *existing == &entry.input_lock && *prior_canonical == canonical,
                    "lock-v2 conflicting input lock pin",
                )?;
            } else {
                require(
                    names.insert(&entry.input_lock.path),
                    "lock-v2 conflicting payload path",
                )?;
                locks.insert(&entry.input_lock.path, (&entry.input_lock, canonical));
                bytes = bytes.checked_add(entry.input_lock.bytes).ok_or_else(|| {
                    ManifestError::Integrity("lock-v2 publication byte count overflow".into())
                })?;
            }
            if let Some(bundle) = &entry.runtime_bundle_manifest {
                bundle.validate()?;
                require(
                    bundle.path.starts_with("runtime-bundles/")
                        && bundle.path.ends_with(".json")
                        && bundle.bytes <= MAX_BUNDLE_BYTES,
                    "lock-v2 publication bundle path/cap",
                )?;
                if let Some(existing) = bundles.get(bundle.path.as_str()) {
                    require(
                        *existing == bundle,
                        "lock-v2 conflicting runtime bundle pin",
                    )?;
                } else {
                    require(
                        names.insert(&bundle.path),
                        "lock-v2 conflicting payload path",
                    )?;
                    bundles.insert(&bundle.path, bundle);
                    bytes = bytes.checked_add(bundle.bytes).ok_or_else(|| {
                        ManifestError::Integrity("lock-v2 publication byte count overflow".into())
                    })?;
                }
            }
        }
        require(
            bytes == self.total_payload_bytes && bytes <= MAX_CORPUS_BYTES,
            "lock-v2 publication total unique payload bytes",
        )?;
        // Match the parser's bound without cloning an entire, potentially
        // attacker-supplied publication into a temporary serialization buffer.
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
            .map_err(|error| ManifestError::Integrity(error.to_string()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(path: &str, byte: char, bytes: u64) -> RoventLockV2PublicationFile {
        RoventLockV2PublicationFile {
            path: path.into(),
            sha256: byte.to_string().repeat(64),
            bytes,
        }
    }

    fn publication() -> RoventLockV2Publication {
        RoventLockV2Publication {
            schema: ROVENT_LOCK_V2_PUBLICATION_DOMAIN.into(),
            manifest: file(MANIFEST_PATH, 'a', 10),
            index: file(INDEX_PATH, 'b', 20),
            chunks: vec![file("chunks/0.jsonl", 'c', 30)],
            canonical_manifest_sha256: "a".repeat(64),
            canonical_index_sha256: "b".repeat(64),
            source_commit: "f".repeat(40),
            source_sha256: "e".repeat(64),
            runtime_sha256: "d".repeat(64),
            record_count: 2,
            total_payload_bytes: 150,
            native_admission_sha256: vec!["a".repeat(64), "b".repeat(64)],
            admissions: ["a", "b"]
                .into_iter()
                .map(|id| RoventLockV2PublicationAdmission {
                    native_admission_sha256: id.repeat(64),
                    input_lock: file("input-locks/lock.json", 'd', 40),
                    input_lock_bytes_origin:
                        RoventLockV2InputLockBytesOrigin::ValidatedSemanticSnapshot,
                    canonical_input_lock_sha256: "e".repeat(64),
                    runtime_bundle_manifest: Some(file("runtime-bundles/runtime.json", 'f', 50)),
                })
                .collect(),
        }
    }

    #[test]
    fn strict_domain_decode_and_unique_shared_payloads() {
        let value = publication();
        let json = serde_json::to_string(&value).unwrap();
        assert_eq!(RoventLockV2Publication::from_json(&json).unwrap(), value);
        let duplicate = json.replacen("\"schema\":", "\"schema\":\"extra\",\"schema\":", 1);
        assert!(RoventLockV2Publication::from_json(&duplicate).is_err());
        let mut old = value;
        old.schema = "rz-rovent-corpus-publication/2".into();
        assert!(old.validate().is_err());
    }

    #[test]
    fn raw_file_pin_is_distinct_from_canonical_lock_and_encoding() {
        let mut value = publication();
        value.admissions[0].input_lock_bytes_origin =
            RoventLockV2InputLockBytesOrigin::OriginalFile;
        value.admissions[1].input_lock = file("input-locks/alternate.json", 'c', 41);
        value.total_payload_bytes += 41;
        assert!(value.validate().is_ok());
        value.admissions[1].input_lock.path = value.admissions[0].input_lock.path.clone();
        assert!(value.validate().is_err());
    }

    #[test]
    fn same_raw_lock_pin_cannot_declare_different_canonical_hashes() {
        let mut value = publication();
        value.admissions[1].canonical_input_lock_sha256 = "a".repeat(64);
        assert!(value.validate().is_err());
        value.admissions[1].input_lock.path = "input-locks/other.json".into();
        value.total_payload_bytes += 40;
        assert!(value.validate().is_err());
    }

    #[test]
    fn rejects_wrong_coverage_paths_and_payload_accounting() {
        let mut value = publication();
        value.admissions[1].native_admission_sha256 = "c".repeat(64);
        assert!(value.validate().is_err());
        let mut value = publication();
        value.chunks[0].path = "chunks/../index.jsonl".into();
        assert!(value.validate().is_err());
        let mut value = publication();
        value.admissions[1]
            .runtime_bundle_manifest
            .as_mut()
            .unwrap()
            .bytes += 1;
        assert!(value.validate().is_err());
        let mut value = publication();
        value.total_payload_bytes += 1;
        assert!(value.validate().is_err());
    }

    #[test]
    fn rejects_unknown_origin_and_oversized_or_unknown_fields() {
        let value = publication();
        let mut json = serde_json::to_value(&value).unwrap();
        json["admissions"][0]["input_lock_bytes_origin"] = serde_json::json!("unverified");
        assert!(RoventLockV2Publication::from_json(&json.to_string()).is_err());
        let mut json = serde_json::to_value(&value).unwrap();
        json["admissions"][0]["surprise"] = serde_json::json!(1);
        assert!(RoventLockV2Publication::from_json(&json.to_string()).is_err());
        let mut value = value;
        value.admissions[0]
            .runtime_bundle_manifest
            .as_mut()
            .unwrap()
            .bytes = MAX_BUNDLE_BYTES + 1;
        assert!(value.validate().is_err());
    }
}
