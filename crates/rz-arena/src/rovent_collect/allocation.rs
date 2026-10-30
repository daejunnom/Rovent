//! 원 allocation 시각의 별도 보조 artifact다. 기존 publication/1 및 raw/2
//! bytes는 수정하지 않고 원 cap이 제공한 canonical bytes만 보존한다.
use super::*;
use serde::{Deserialize, Serialize};
pub const ROVENT_ALLOCATION_SUPPLEMENT: &str = "rz-rovent-corpus-allocation-supplement/1";
pub const ROVENT_R_ALLOCATION_SUPPLEMENT: &str = "rz-rovent-r-corpus-allocation-supplement/1";
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventAllocationSupplementEntry {
    pub native_admission_sha256: String,
    pub root_descriptor_sha256: String,
    pub allocation: RoventPublicationFile,
    pub canonical_allocation_sha256: String,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventAllocationSupplement {
    pub schema: String,
    pub base_publication: RoventPublicationFile,
    pub canonical_manifest_sha256: String,
    pub canonical_index_sha256: String,
    pub entries: Vec<RoventAllocationSupplementEntry>,
    pub allocation_payload_bytes: u64,
}
impl RoventAllocationSupplement {
    pub fn validate(&self) -> Result<(), ArenaError> {
        self.base_publication.validate()?;
        check(
            ((self.schema == ROVENT_ALLOCATION_SUPPLEMENT
                && self.base_publication.path == "publication.json")
                || (self.schema == ROVENT_R_ALLOCATION_SUPPLEMENT
                    && self.base_publication.path == "r-publication.json"))
                && self.base_publication.bytes <= 16 * 1024 * 1024
                && generation_sha(&self.canonical_manifest_sha256)
                && generation_sha(&self.canonical_index_sha256)
                && !self.entries.is_empty()
                && self.entries.len() <= MAX_CORPUS_ROWS
                && self
                    .entries
                    .windows(2)
                    .all(|v| v[0].native_admission_sha256 < v[1].native_admission_sha256),
            "allocation supplement header/order",
        )?;
        let mut bytes = 0u64;
        let mut paths = BTreeSet::new();
        for e in &self.entries {
            e.allocation.validate()?;
            check(
                [
                    &e.native_admission_sha256,
                    &e.root_descriptor_sha256,
                    &e.canonical_allocation_sha256,
                ]
                .iter()
                .all(|v| generation_sha(v))
                    && e.allocation.path
                        == format!("allocations/{}.json", e.canonical_allocation_sha256)
                    && e.allocation.bytes <= 4096
                    && paths.insert(&e.allocation.path),
                "allocation supplement exact file pin",
            )?;
            bytes = bytes
                .checked_add(e.allocation.bytes)
                .ok_or_else(|| invalid("allocation bytes overflow"))?;
        }
        check(
            bytes == self.allocation_payload_bytes && bytes <= MAX_CORPUS_BYTES,
            "allocation supplement payload bytes",
        )
    }
}
struct CapturedAllocation {
    descriptor: rz_experiments::RoventRootDescriptor,
    // 검증한 원 clock은 capture와 함께 유지한다. supplement 게시 소비자는 Linux 전용이다.
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    clock: rz_experiments::RoventRootAllocationClock,
    raw: Vec<u8>,
}
pub(super) struct RoventAllocationSupplementCapture {
    entries: BTreeMap<String, CapturedAllocation>,
}
impl RoventAllocationSupplementCapture {
    pub(super) fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }
    pub(super) fn retain_admission(
        &mut self,
        admission: &RoventCollectionRootAdmission,
    ) -> Result<u64, ArenaError> {
        let root = admission.root_admission();
        let sha = admission.raw_sha256();
        let raw = root.allocation_clock_canonical_bytes();
        let clock = root.allocation_clock();
        clock.verify_against(
            root.descriptor(),
            admission.native_origin().original_input_lock(),
        )?;
        check(
            clock.canonical_bytes()?.as_slice() == raw
                && clock.digest()? == root.allocation_clock_digest()
                && raw.len() <= 4096,
            "original allocation cap body/digest",
        )?;
        if let Some(old) = self.entries.get(sha) {
            check(
                old.raw == raw && old.descriptor == *root.descriptor(),
                "same root allocation changed",
            )?;
            return Ok(0);
        }
        check(
            self.entries.len() < MAX_CORPUS_ROWS,
            "allocation root count cap",
        )?;
        self.entries.insert(
            sha.into(),
            CapturedAllocation {
                descriptor: root.descriptor().clone(),
                clock: clock.clone(),
                raw: raw.to_vec(),
            },
        );
        Ok(raw.len() as u64 + 512)
    }
    #[cfg(target_os = "linux")]
    pub(super) fn publish(
        &self,
        directory: &cap_std::fs::Dir,
        manifest: &RoventCorpusManifest,
        index: &RoventCorpusIndex,
        until: std::time::Instant,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> Result<RoventAllocationSupplement, ArenaError> {
        self.publish_base(
            directory,
            "publication.json",
            manifest.sha256()?,
            index.sha256()?,
            index.record_count,
            false,
            until,
            cancel,
        )
    }
    #[cfg(target_os = "linux")]
    pub(super) fn publish_r(
        &self,
        directory: &cap_std::fs::Dir,
        manifest: &rz_experiments::rovent_r_readout::RoventRReadoutManifest,
        index: &rz_experiments::rovent_r_readout::RoventRReadoutIndex,
        until: std::time::Instant,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> Result<RoventAllocationSupplement, ArenaError> {
        self.publish_base(
            directory,
            "r-publication.json",
            manifest.sha256()?,
            index.sha256()?,
            index.record_count,
            false,
            until,
            cancel,
        )
    }
    #[cfg(target_os = "linux")]
    pub(super) fn publish_submitted(
        &self,
        directory: &cap_std::fs::Dir,
        manifest: &rz_experiments::rovent_corpus_v2::RoventCorpusManifestV2,
        index: &rz_experiments::rovent_corpus_v2::RoventCorpusIndexV2,
        until: std::time::Instant,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> Result<RoventAllocationSupplement, ArenaError> {
        self.publish_base(
            directory,
            "publication.json",
            manifest.sha256()?,
            index.sha256()?,
            index.record_count,
            true,
            until,
            cancel,
        )
    }
    #[cfg(target_os = "linux")]
    #[allow(clippy::too_many_arguments)]
    fn publish_base(
        &self,
        directory: &cap_std::fs::Dir,
        base_path: &str,
        manifest_sha: String,
        index_sha: String,
        record_count: u64,
        submitted_family: bool,
        until: std::time::Instant,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> Result<RoventAllocationSupplement, ArenaError> {
        use cap_fs_ext::{FollowSymlinks, OpenOptionsFollowExt};
        use std::io::{Read, Write};
        let io = |e: std::io::Error| ArenaError::Io(e.to_string());
        let active = || {
            check(
                !cancel.load(std::sync::atomic::Ordering::Acquire)
                    && std::time::Instant::now() < until,
                "allocation supplement canceled/expired",
            )
        };
        active()?;
        let mut options = cap_std::fs::OpenOptions::new();
        options.read(true).follow(FollowSymlinks::No);
        let mut file = directory.open_with(base_path, &options).map_err(io)?;
        let meta = file.metadata().map_err(io)?;
        check(
            meta.is_file() && meta.len() <= 16 * 1024 * 1024,
            "allocation original base publication type/cap",
        )?;
        let mut raw = Vec::new();
        Read::take(&mut file, 16 * 1024 * 1024 + 1)
            .read_to_end(&mut raw)
            .map_err(io)?;
        check(
            raw.len() as u64 == meta.len(),
            "allocation base publication changed bytes",
        )?;
        let text = std::str::from_utf8(&raw).map_err(|_| invalid("base publication UTF8"))?;
        let (base_manifest, base_index, base_count, base_admissions, base_bytes) =
            if base_path == "publication.json" && submitted_family {
                let base: RoventCorpusPublicationV2 = rz_experiments::decode_json(text)?;
                base.validate()?;
                (
                    base.canonical_manifest_sha256,
                    base.canonical_index_sha256,
                    base.record_count,
                    base.native_admission_sha256,
                    base.total_payload_bytes,
                )
            } else if base_path == "publication.json" {
                let base: RoventCorpusPublication = rz_experiments::decode_json(text)?;
                base.validate()?;
                (
                    base.canonical_manifest_sha256,
                    base.canonical_index_sha256,
                    base.record_count,
                    base.native_admission_sha256,
                    base.total_payload_bytes,
                )
            } else {
                let base: super::r_publication::RoventRCorpusPublication =
                    rz_experiments::decode_json(text)?;
                base.validate()?;
                (
                    base.canonical_manifest_sha256,
                    base.canonical_index_sha256,
                    base.record_count,
                    base.admissions
                        .into_iter()
                        .map(|a| a.native_admission_sha256)
                        .collect(),
                    base.total_payload_bytes,
                )
            };
        check(
            base_manifest == manifest_sha
                && base_index == index_sha
                && base_count == record_count
                && base_admissions.iter().eq(self.entries.keys()),
            "allocation exact original corpus/admission coverage",
        )?;
        let base_publication = RoventPublicationFile {
            path: base_path.into(),
            sha256: hex(Digest(Sha256::digest(&raw).into())),
            bytes: raw.len() as u64,
        };
        directory.create_dir("allocations").map_err(io)?;
        let put = |path: &str, bytes: &[u8]| -> Result<RoventPublicationFile, ArenaError> {
            active()?;
            let mut options = cap_std::fs::OpenOptions::new();
            options
                .write(true)
                .create_new(true)
                .follow(FollowSymlinks::No);
            let mut file = directory.open_with(path, &options).map_err(io)?;
            file.write_all(bytes)
                .and_then(|_| file.sync_all())
                .map_err(io)?;
            Ok(RoventPublicationFile {
                path: path.into(),
                sha256: hex(Digest(Sha256::digest(bytes).into())),
                bytes: bytes.len() as u64,
            })
        };
        let mut entries = Vec::new();
        let mut allocation_payload_bytes = 0;
        for (sha, captured) in &self.entries {
            active()?;
            let canonical = hex(captured.clock.digest()?);
            let allocation = put(&format!("allocations/{canonical}.json"), &captured.raw)?;
            allocation_payload_bytes += allocation.bytes;
            entries.push(RoventAllocationSupplementEntry {
                native_admission_sha256: sha.clone(),
                root_descriptor_sha256: hex(captured.descriptor.digest()?),
                allocation,
                canonical_allocation_sha256: canonical,
            });
        }
        let supplement = RoventAllocationSupplement {
            schema: if base_path == "publication.json" {
                ROVENT_ALLOCATION_SUPPLEMENT
            } else {
                ROVENT_R_ALLOCATION_SUPPLEMENT
            }
            .into(),
            base_publication,
            canonical_manifest_sha256: manifest_sha,
            canonical_index_sha256: index_sha,
            entries,
            allocation_payload_bytes,
        };
        supplement.validate()?;
        let raw = serde_json::to_vec(&supplement).map_err(|e| invalid(e.to_string()))?;
        check(
            raw.len() <= 16 * 1024 * 1024
                && base_bytes
                    .checked_add(supplement.allocation_payload_bytes)
                    .and_then(|n| n.checked_add(raw.len() as u64))
                    .is_some_and(|n| n <= MAX_CORPUS_BYTES),
            "allocation supplement metadata/aggregate corpus cap",
        )?;
        put("allocation-supplement.pending.json", &raw)?;
        let sub = directory.open_dir("allocations").map_err(io)?;
        crate::pals_collect_v3::training_process::directory_file(&sub)?
            .sync_all()
            .map_err(io)?;
        let root = crate::pals_collect_v3::training_process::directory_file(directory)?;
        root.sync_all().map_err(io)?;
        active()?;
        directory
            .hard_link(
                "allocation-supplement.pending.json",
                directory,
                "allocation-supplement.json",
            )
            .map_err(io)?;
        directory
            .remove_file("allocation-supplement.pending.json")
            .map_err(io)?;
        root.sync_all().map_err(io)?;
        Ok(supplement)
    }
}
