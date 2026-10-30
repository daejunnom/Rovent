//! 실제 writer가 마지막에 게시할 내구성 commit. 이 문서의 무결성 확인은
//! historical data 검증이며 native 실행·물리 완료·학습 권한을 복원하지 않는다.
use super::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub const ROVENT_CORPUS_PUBLICATION: &str = "rz-rovent-corpus-publication/1";
pub const ROVENT_PUBLICATION_FILE: &str = "publication.json";
pub const ROVENT_PUBLICATION_MANIFEST: &str = "corpus-manifest.json";
pub const ROVENT_PUBLICATION_INDEX: &str = "corpus-index.json";

/// publication root 기준 normalized 상대 경로다. 원 source/license가 필요한
/// 외부 ArtifactRef와 달리 이 값은 이미 게시한 파일의 raw bytes pin만 담는다.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventPublicationFile {
    pub path: String,
    pub sha256: String,
    pub bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    fn pin(path: &str, hash: char, bytes: u64) -> RoventPublicationFile {
        RoventPublicationFile {
            path: path.into(),
            sha256: hash.to_string().repeat(64),
            bytes,
        }
    }
    fn metadata_only() -> RoventCorpusPublication {
        let lock = pin("input-locks/lock.json", 'd', 40);
        let bundle = pin("runtime-bundles/bundle.json", 'e', 50);
        RoventCorpusPublication {
            schema: ROVENT_CORPUS_PUBLICATION.into(),
            manifest: pin(ROVENT_PUBLICATION_MANIFEST, 'a', 10),
            index: pin(ROVENT_PUBLICATION_INDEX, 'b', 20),
            chunks: vec![pin("chunks/000000.jsonl", 'c', 30)],
            canonical_manifest_sha256: "a".repeat(64),
            canonical_index_sha256: "b".repeat(64),
            source_commit: "f".repeat(40),
            source_sha256: "f".repeat(64),
            runtime_sha256: "a".repeat(64),
            record_count: 2,
            total_payload_bytes: 150,
            native_admission_sha256: vec!["a".repeat(64), "b".repeat(64)],
            admissions: ["a", "b"]
                .map(|sha| RoventPublicationAdmission {
                    native_admission_sha256: sha.repeat(64),
                    input_lock: lock.clone(),
                    canonical_input_lock_sha256: "d".repeat(64),
                    runtime_bundle_manifest: Some(bundle.clone()),
                })
                .into(),
        }
    }
    #[test]
    fn publication_shape_counts_shared_original_metadata_once() {
        // DTO shape only: no original native, writer or historical owner exists.
        let publication = metadata_only();
        publication.validate().unwrap();
        let decoded: RoventCorpusPublication =
            serde_json::from_slice(&serde_json::to_vec(&publication).unwrap()).unwrap();
        assert_eq!(publication, decoded);
    }
    #[test]
    fn publication_shape_rejects_uncovered_or_conflicting_origin_metadata() {
        let mut publication = metadata_only();
        publication.admissions[1].native_admission_sha256 = "c".repeat(64);
        assert!(publication.validate().is_err());
        let mut publication = metadata_only();
        publication.admissions[1].input_lock.path = "input-locks/other.json".into();
        assert!(publication.validate().is_err());
        let mut publication = metadata_only();
        publication.admissions[1]
            .runtime_bundle_manifest
            .as_mut()
            .unwrap()
            .bytes += 1;
        assert!(publication.validate().is_err());
    }
    #[test]
    fn publication_shape_rejects_path_escape_and_inexact_total() {
        let mut publication = metadata_only();
        publication.chunks[0].path = "chunks/../elsewhere.jsonl".into();
        assert!(publication.validate().is_err());
        let mut publication = metadata_only();
        publication.total_payload_bytes += 1;
        assert!(publication.validate().is_err());
    }
}
impl RoventPublicationFile {
    pub fn validate(&self) -> Result<(), ArenaError> {
        check(
            relative_file(&self.path)
                && generation_sha(&self.sha256)
                && self.bytes > 0
                && self.bytes <= MAX_CORPUS_BYTES,
            "publication normalized relative file pin",
        )
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

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventCorpusPublication {
    pub schema: String,
    pub manifest: RoventPublicationFile,
    pub index: RoventPublicationFile,
    pub chunks: Vec<RoventPublicationFile>,
    pub canonical_manifest_sha256: String,
    pub canonical_index_sha256: String,
    pub source_commit: String,
    pub source_sha256: String,
    /// 원 loaded runtime library 파일의 SHA다. 전체 구성 aggregate가 아니다.
    pub runtime_sha256: String,
    pub record_count: u64,
    pub total_payload_bytes: u64,
    pub native_admission_sha256: Vec<String>,
    pub admissions: Vec<RoventPublicationAdmission>,
}

/// 각 원 root admission이 실제 loaded lock snapshot을 지목한다. 같은 원
/// loaded lock을 공유하는 root는 같은 exact file pin을 재사용한다.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventPublicationAdmission {
    pub native_admission_sha256: String,
    pub input_lock: RoventPublicationFile,
    pub canonical_input_lock_sha256: String,
    /// Exact original JSON file pin; its file SHA is not the parsed CUDA
    /// binary-codec/loader-aliases digest in the native origin.
    pub runtime_bundle_manifest: Option<RoventPublicationFile>,
}
impl RoventCorpusPublication {
    /// DTO shape 검사다. 파일 pin·전체 record·Rules replay·target·계보 검사나
    /// 원 writer cap을 대신하지 않는다.
    pub fn validate(&self) -> Result<(), ArenaError> {
        self.manifest.validate()?;
        self.index.validate()?;
        check(
            self.schema == ROVENT_CORPUS_PUBLICATION
                && self.manifest.path == ROVENT_PUBLICATION_MANIFEST
                && self.index.path == ROVENT_PUBLICATION_INDEX
                && self.source_commit.len() == 40
                && self
                    .source_commit
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
                && [
                    &self.canonical_manifest_sha256,
                    &self.canonical_index_sha256,
                    &self.source_sha256,
                    &self.runtime_sha256,
                ]
                .into_iter()
                .all(|sha| generation_sha(sha))
                && self.record_count > 0
                && self.record_count <= MAX_CORPUS_ROWS as u64
                && !self.chunks.is_empty()
                && self.chunks.len() <= 8192
                && !self.native_admission_sha256.is_empty()
                && self.native_admission_sha256.len() <= MAX_CORPUS_ROWS
                && self
                    .native_admission_sha256
                    .iter()
                    .all(|sha| generation_sha(sha))
                && self
                    .native_admission_sha256
                    .windows(2)
                    .all(|pair| pair[0] < pair[1]),
            "publication schema/source/count/admission seals",
        )?;
        check(
            self.admissions.len() == self.native_admission_sha256.len()
                && self
                    .admissions
                    .iter()
                    .zip(&self.native_admission_sha256)
                    .all(|(entry, sha)| {
                        &entry.native_admission_sha256 == sha
                            && generation_sha(&entry.canonical_input_lock_sha256)
                    }),
            "publication exact sorted admission/lock coverage",
        )?;
        let mut names = BTreeSet::from([self.manifest.path.as_str(), self.index.path.as_str()]);
        let mut bytes = self
            .manifest
            .bytes
            .checked_add(self.index.bytes)
            .ok_or_else(|| invalid("publication byte count overflow"))?;
        for chunk in &self.chunks {
            chunk.validate()?;
            check(
                chunk.path.starts_with("chunks/")
                    && chunk.path.ends_with(".jsonl")
                    && chunk.bytes <= MAX_CHUNK_BYTES as u64
                    && names.insert(&chunk.path),
                "publication chunk path/cap/duplicate",
            )?;
            bytes = bytes
                .checked_add(chunk.bytes)
                .ok_or_else(|| invalid("publication byte count overflow"))?;
        }
        let mut locks: BTreeMap<&str, &RoventPublicationFile> = BTreeMap::new();
        let mut bundles: BTreeMap<&str, &RoventPublicationFile> = BTreeMap::new();
        for entry in &self.admissions {
            entry.input_lock.validate()?;
            check(
                entry.input_lock.path.starts_with("input-locks/")
                    && entry.input_lock.path.ends_with(".json"),
                "publication input lock path",
            )?;
            if let Some(existing) = locks.get(entry.canonical_input_lock_sha256.as_str()) {
                check(
                    *existing == &entry.input_lock,
                    "one canonical lock has different file pins",
                )?;
            } else {
                check(
                    names.insert(&entry.input_lock.path),
                    "publication conflicting lock path",
                )?;
                locks.insert(&entry.canonical_input_lock_sha256, &entry.input_lock);
                bytes = bytes
                    .checked_add(entry.input_lock.bytes)
                    .ok_or_else(|| invalid("publication byte count overflow"))?;
            }
            if let Some(bundle) = &entry.runtime_bundle_manifest {
                bundle.validate()?;
                check(
                    bundle.path.starts_with("runtime-bundles/")
                        && bundle.path.ends_with(".json")
                        && bundle.bytes <= 64 * 1024,
                    "publication runtime bundle metadata path/cap",
                )?;
                if let Some(existing) = bundles.get(bundle.sha256.as_str()) {
                    check(
                        *existing == bundle,
                        "one bundle file SHA has different file pins",
                    )?;
                } else {
                    check(
                        names.insert(&bundle.path),
                        "publication conflicting runtime bundle path",
                    )?;
                    bundles.insert(&bundle.sha256, bundle);
                    bytes = bytes
                        .checked_add(bundle.bytes)
                        .ok_or_else(|| invalid("publication byte count overflow"))?;
                }
            }
        }
        check(
            bytes == self.total_payload_bytes && bytes <= MAX_CORPUS_BYTES,
            "publication exact payload byte total",
        )
    }
}

/// finish_publication에서만 만들며 metadata decoder는 이 타입을 발급하지 않는다.
pub struct FinishedRoventCorpus {
    manifest: RoventCorpusManifest,
    index: RoventCorpusIndex,
    native_admission_sha256: Vec<String>,
    // 원 출처 snapshot의 소유권은 유지한다. durable 게시 소비자만 Linux 전용이다.
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    admissions: BTreeMap<String, Arc<RoventPublicationSourceSnapshot>>,
    // 비-Linux 완료 owner도 원 allocation 근거를 버리지 않고 보유한다.
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    allocations: super::allocation::RoventAllocationSupplementCapture,
}

/// Minted only while consuming an original PublishedRecord. It deliberately
/// retains metadata, not every old Rules root or the loaded model backing.
pub(super) struct RoventPublicationSourceSnapshot {
    lock: rz_experiments::RoventInputLock,
    lock_bytes: Vec<u8>,
    bundle: Option<(rz_experiments::ArtifactRef, Vec<u8>)>,
}
impl RoventPublicationSourceSnapshot {
    pub(super) fn capture(admission: &RoventCollectionRootAdmission) -> Result<Self, ArenaError> {
        let origin = admission.native_origin();
        let lock = origin.original_input_lock();
        lock.validate()?;
        check(
            origin.raw_receipt()["input_lock_sha256"] == lock.canonical_sha256,
            "original loaded lock snapshot differs from native origin",
        )?;
        let lock_bytes = serde_json::to_vec(lock).map_err(|error| invalid(error.to_string()))?;
        check(
            lock_bytes.len() <= 16 * 1024 * 1024,
            "original lock metadata cap",
        )?;
        let bundle = origin
            .runtime_bundle_publication_bytes()
            .map_err(|error| invalid(error.to_string()))?;
        Ok(Self {
            lock: lock.clone(),
            lock_bytes,
            bundle,
        })
    }
    pub(super) fn same_source(&self, other: &Self) -> bool {
        self.lock == other.lock
            && self.lock_bytes == other.lock_bytes
            && self.bundle == other.bundle
    }
    pub(super) fn payload_key(&self) -> String {
        let mut hash = Sha256::new();
        hash.update(b"rz-rovent-publication-source-metadata/1\0");
        hash.update((self.lock_bytes.len() as u64).to_le_bytes());
        hash.update(&self.lock_bytes);
        if let Some((pin, _)) = &self.bundle {
            hash.update([1]);
            hash.update(pin.sha256.as_bytes());
            hash.update(pin.bytes.to_le_bytes());
        } else {
            hash.update([0]);
        }
        hex(Digest(hash.finalize().into()))
    }
    pub(super) fn payload_bytes(&self) -> Result<u64, ArenaError> {
        (self.lock_bytes.len() as u64)
            .checked_add(
                self.bundle
                    .as_ref()
                    .map_or(0, |(_, bytes)| bytes.len() as u64),
            )
            .ok_or_else(|| invalid("publication source metadata byte overflow"))
    }
}

// The durable file writer shares only the already captured metadata bytes.
// Implementing this trait cannot create a live admission or publication proof.
#[cfg(target_os = "linux")]
pub(super) trait PublicationSourceBytes {
    fn canonical_lock_sha256(&self) -> &str;
    fn lock_bytes(&self) -> &[u8];
    fn runtime_bundle(&self) -> Option<(&rz_experiments::ArtifactRef, &[u8])>;
}

#[cfg(target_os = "linux")]
impl PublicationSourceBytes for RoventPublicationSourceSnapshot {
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
impl FinishedRoventCorpus {
    pub(super) fn from_writer(writer: RoventCorpusWriter) -> Result<Self, ArenaError> {
        check(
            writer.buffer.is_empty() && !writer.admissions.is_empty(),
            "flush original actual corpus before finish publication",
        )?;
        writer.index.validate()?;
        check(
            writer.index.record_count > 0
                && writer.index.record_count as usize == writer.record_seals.len(),
            "finished actual corpus is empty or incomplete",
        )?;
        for bytes in [
            serde_json::to_vec(&writer.manifest),
            serde_json::to_vec(&writer.index),
        ] {
            check(
                bytes.map_err(|error| invalid(error.to_string()))?.len() <= 16 * 1024 * 1024,
                "finished publication manifest/index metadata cap",
            )?;
        }
        Ok(Self {
            manifest: writer.manifest,
            index: writer.index,
            native_admission_sha256: writer.admissions.keys().cloned().collect(),
            admissions: writer.admissions,
            allocations: writer.allocations,
        })
    }
    pub fn manifest(&self) -> &RoventCorpusManifest {
        &self.manifest
    }
    pub fn index(&self) -> &RoventCorpusIndex {
        &self.index
    }
    pub fn native_admission_sha256(&self) -> &[String] {
        &self.native_admission_sha256
    }
    /// 원 /1 commit과 raw /2를 변경하지 않는 별도 시각 근거 artifact다.
    /// base publication을 먼저 게시한 뒤 같은 private directory에서 호출한다.
    #[cfg(target_os = "linux")]
    pub fn publish_allocation_supplement(
        &self,
        directory: &cap_std::fs::Dir,
        until: std::time::Instant,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> Result<super::allocation::RoventAllocationSupplement, ArenaError> {
        self.allocations
            .publish(directory, &self.manifest, &self.index, until, cancel)
    }

    /// Linux durable publication over the caller's original private directory
    /// capability. Chunks must already exist there. All payloads are checked
    /// and synced before the final no-overwrite publication commit is exposed.
    /// A partial failure preserves its files; it never grants a live capability.
    #[cfg(target_os = "linux")]
    pub fn publish_atomic(
        &self,
        directory: &cap_std::fs::Dir,
        until: std::time::Instant,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> Result<RoventCorpusPublication, ArenaError> {
        let manifest_bytes =
            serde_json::to_vec(&self.manifest).map_err(|error| invalid(error.to_string()))?;
        let index_bytes =
            serde_json::to_vec(&self.index).map_err(|error| invalid(error.to_string()))?;
        publish_original_payloads(
            directory,
            until,
            cancel,
            RoventPublicationMaterial {
                manifest_bytes: &manifest_bytes,
                index_bytes: &index_bytes,
                chunks: &self.index.chunks,
                admissions: &self.admissions,
            },
            |manifest, index, chunks, admissions, payload_bytes| {
                let publication = RoventCorpusPublication {
                    schema: ROVENT_CORPUS_PUBLICATION.into(),
                    manifest,
                    index,
                    chunks,
                    canonical_manifest_sha256: self.manifest.sha256()?,
                    canonical_index_sha256: self.index.sha256()?,
                    source_commit: self.manifest.source_commit.clone(),
                    source_sha256: self.manifest.source_sha256.clone(),
                    runtime_sha256: self.manifest.runtime_sha256.clone(),
                    record_count: self.index.record_count,
                    total_payload_bytes: payload_bytes,
                    native_admission_sha256: self.native_admission_sha256.clone(),
                    admissions,
                };
                publication.validate()?;
                Ok(publication)
            },
        )
    }
}

/// 기존 내구성 I/O만 공유한다. 원 writer의 private admission source와 형식별
/// validator를 요구하며 공개 metadata를 live 권한으로 되살리지 않는다.
#[cfg(target_os = "linux")]
pub(super) struct RoventPublicationMaterial<'a, S> {
    pub manifest_bytes: &'a [u8],
    pub index_bytes: &'a [u8],
    pub chunks: &'a [RoventCorpusChunk],
    pub admissions: &'a BTreeMap<String, Arc<S>>,
}
#[cfg(target_os = "linux")]
pub(super) fn publish_original_payloads<P: serde::Serialize, S: PublicationSourceBytes>(
    directory: &cap_std::fs::Dir,
    until: std::time::Instant,
    cancel: &std::sync::atomic::AtomicBool,
    material: RoventPublicationMaterial<'_, S>,
    build: impl FnOnce(
        RoventPublicationFile,
        RoventPublicationFile,
        Vec<RoventPublicationFile>,
        Vec<RoventPublicationAdmission>,
        u64,
    ) -> Result<P, ArenaError>,
) -> Result<P, ArenaError> {
    publish_original_payloads_authorized(directory, until, cancel, material, build, || Ok(()))
}

/// Optional final authorization runs after all prepared bytes have been
/// synced and immediately before the publication hard-link. Failure leaves
/// the original pending files; it never creates a success receipt.
#[cfg(target_os = "linux")]
pub(super) fn publish_original_payloads_authorized<P: serde::Serialize, S: PublicationSourceBytes>(
    directory: &cap_std::fs::Dir,
    until: std::time::Instant,
    cancel: &std::sync::atomic::AtomicBool,
    material: RoventPublicationMaterial<'_, S>,
    build: impl FnOnce(
        RoventPublicationFile,
        RoventPublicationFile,
        Vec<RoventPublicationFile>,
        Vec<RoventPublicationAdmission>,
        u64,
    ) -> Result<P, ArenaError>,
    authorize: impl FnOnce() -> Result<(), ArenaError>,
) -> Result<P, ArenaError> {
    use cap_fs_ext::{FollowSymlinks, OpenOptionsFollowExt};
    use std::io::Read;
    fn io(error: std::io::Error) -> ArenaError {
        ArenaError::Io(error.to_string())
    }
    let active = || -> Result<(), ArenaError> {
        check(
            !cancel.load(std::sync::atomic::Ordering::Acquire) && std::time::Instant::now() < until,
            "publication canceled/expired; partial output preserved",
        )
    };
    active()?;
    fn put(
        directory: &cap_std::fs::Dir,
        path: &str,
        bytes: &[u8],
    ) -> Result<RoventPublicationFile, ArenaError> {
        use cap_fs_ext::{FollowSymlinks, OpenOptionsFollowExt};
        use std::io::Write;
        check(
            relative_file(path) && !bytes.is_empty(),
            "publication owned relative payload",
        )?;
        let mut options = cap_std::fs::OpenOptions::new();
        options
            .write(true)
            .create_new(true)
            .follow(FollowSymlinks::No);
        let mut file = directory.open_with(path, &options).map_err(io)?;
        file.write_all(bytes)
            .and_then(|_| file.sync_all())
            .map_err(io)?;
        check(
            file.metadata().map_err(io)?.is_file(),
            "publication payload is not regular",
        )?;
        Ok(RoventPublicationFile {
            path: path.into(),
            sha256: hex(Digest(Sha256::digest(bytes).into())),
            bytes: bytes.len() as u64,
        })
    }
    let mut payload_bytes =
        material.manifest_bytes.len() as u64 + material.index_bytes.len() as u64;
    let mut chunks = Vec::with_capacity(material.chunks.len());
    for chunk in material.chunks {
        active()?;
        let pin = RoventPublicationFile {
            path: chunk.path.clone(),
            sha256: chunk.sha256.clone(),
            bytes: chunk.bytes,
        };
        pin.validate()?;
        let mut options = cap_std::fs::OpenOptions::new();
        options.read(true).follow(FollowSymlinks::No);
        let mut file = directory.open_with(&pin.path, &options).map_err(io)?;
        let metadata = file.metadata().map_err(io)?;
        check(
            metadata.is_file()
                && metadata.len() == pin.bytes
                && pin.bytes <= MAX_CHUNK_BYTES as u64,
            "original publication chunk type/size",
        )?;
        let mut hash = Sha256::new();
        let mut bounded = Read::take(&mut file, pin.bytes + 1);
        let mut buffer = [0u8; 64 * 1024];
        let mut copied = 0u64;
        loop {
            active()?;
            let count = bounded.read(&mut buffer).map_err(io)?;
            if count == 0 {
                break;
            }
            copied = copied
                .checked_add(count as u64)
                .ok_or_else(|| invalid("publication chunk read overflow"))?;
            hash.update(&buffer[..count]);
        }
        check(
            copied == pin.bytes && hex(Digest(hash.finalize().into())) == pin.sha256,
            "original publication chunk bytes changed",
        )?;
        file.sync_all().map_err(io)?;
        payload_bytes = payload_bytes
            .checked_add(pin.bytes)
            .ok_or_else(|| invalid("publication payload overflow"))?;
        chunks.push(pin);
    }
    directory.create_dir("input-locks").map_err(io)?;
    if material
        .admissions
        .values()
        .any(|source| source.runtime_bundle().is_some())
    {
        directory.create_dir("runtime-bundles").map_err(io)?;
    }
    let mut locks: BTreeMap<String, RoventPublicationFile> = BTreeMap::new();
    let mut bundles: BTreeMap<String, RoventPublicationFile> = BTreeMap::new();
    let mut admissions = Vec::with_capacity(material.admissions.len());
    for (native_sha, source) in material.admissions {
        active()?;
        let canonical = source.canonical_lock_sha256();
        let input_lock = match locks.get(canonical) {
            Some(pin) => pin.clone(),
            None => {
                let pin = put(
                    directory,
                    &format!("input-locks/{canonical}.json"),
                    source.lock_bytes(),
                )?;
                payload_bytes = payload_bytes
                    .checked_add(pin.bytes)
                    .ok_or_else(|| invalid("publication payload overflow"))?;
                locks.insert(canonical.to_owned(), pin.clone());
                pin
            }
        };
        let runtime_bundle_manifest = if let Some((original, bytes)) = source.runtime_bundle() {
            Some(match bundles.get(&original.sha256) {
                Some(pin) => pin.clone(),
                None => {
                    let pin = put(
                        directory,
                        &format!("runtime-bundles/{}.json", original.sha256),
                        bytes,
                    )?;
                    check(
                        pin.sha256 == original.sha256 && pin.bytes == original.bytes,
                        "original runtime bundle publication bytes changed",
                    )?;
                    payload_bytes = payload_bytes
                        .checked_add(pin.bytes)
                        .ok_or_else(|| invalid("publication payload overflow"))?;
                    bundles.insert(original.sha256.clone(), pin.clone());
                    pin
                }
            })
        } else {
            None
        };
        admissions.push(RoventPublicationAdmission {
            native_admission_sha256: native_sha.clone(),
            input_lock,
            canonical_input_lock_sha256: canonical.to_owned(),
            runtime_bundle_manifest,
        });
    }
    check(payload_bytes <= MAX_CORPUS_BYTES, "publication output cap")?;
    let manifest = put(
        directory,
        ROVENT_PUBLICATION_MANIFEST,
        material.manifest_bytes,
    )?;
    active()?;
    let index = put(directory, ROVENT_PUBLICATION_INDEX, material.index_bytes)?;
    let publication = build(manifest, index, chunks, admissions, payload_bytes)?;
    let raw = serde_json::to_vec(&publication).map_err(|error| invalid(error.to_string()))?;
    check(
        raw.len() <= 16 * 1024 * 1024,
        "publication commit metadata cap",
    )?;
    let pending = "publication.pending.json";
    active()?;
    put(directory, pending, &raw)?;
    let original_directory = crate::pals_collect_v3::training_process::directory_file(directory)?;
    for name in ["chunks", "input-locks", "runtime-bundles"] {
        if name == "runtime-bundles" && bundles.is_empty() {
            continue;
        }
        let subdirectory = directory.open_dir(name).map_err(io)?;
        crate::pals_collect_v3::training_process::directory_file(&subdirectory)?
            .sync_all()
            .map_err(io)?;
    }
    original_directory.sync_all().map_err(io)?;
    active()?;
    authorize()?;
    directory
        .hard_link(pending, directory, ROVENT_PUBLICATION_FILE)
        .map_err(io)?;
    directory.remove_file(pending).map_err(io)?;
    original_directory.sync_all().map_err(io)?;
    Ok(publication)
}
