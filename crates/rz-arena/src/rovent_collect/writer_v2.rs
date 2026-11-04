//! raw/3 원 PublishedRecord만 받는 별도 writer다. 원 유한 chunk/출처 보존과
//! durable 파일 I/O는 공통 경로를 사용하며 /1 artifact로 재서명하지 않는다.
use super::*;
use rz_experiments::rovent_corpus_v2::{RoventCorpusIndexV2, RoventCorpusManifestV2};
use rz_experiments::rovent_submitted_input::ROVENT_INDEX_V2_DOMAIN;
use std::sync::Arc;

pub struct RoventCorpusWriterV2 {
    manifest: RoventCorpusManifestV2,
    original: RoventCorpusWriter,
}
pub(super) struct RoventWriterFinishFailureV2 {
    pub failure: ArenaError,
    // 실패 객체의 원 writer 소유권은 유지한다. 복구 소비자는 Linux launch에 있다.
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    pub writer: RoventCorpusWriterV2,
}
impl RoventCorpusWriterV2 {
    pub fn new(manifest: &RoventCorpusManifestV2) -> Result<Self, ArenaError> {
        let mut original = RoventCorpusWriter::new(&manifest.common_registration_projection())?;
        original.index.manifest_sha256 = manifest.sha256()?;
        Ok(Self {
            manifest: manifest.clone(),
            original,
        })
    }
    pub fn append(&mut self, row: PublishedRoventRecordV2) -> Result<(), ArenaError> {
        self.append_original(&row)
    }
    /// Caller는 실패한 원 PublishedRecord를 보존한다. 성공 뒤에만 caller가
    /// 버리며 중복 게시 검사는 같은 writer의 원 record seal 집합이 수행한다.
    pub(super) fn append_original(
        &mut self,
        row: &PublishedRoventRecordV2,
    ) -> Result<(), ArenaError> {
        check(
            row.manifest_sha256 == self.original.index.manifest_sha256,
            "raw3 writer original manifest mismatch",
        )?;
        row.record.validate(&self.manifest)?;
        check(
            row.record.native.raw_retirement["root_admission_sha256"] == row.admission.raw_sha256()
                && row.record.native.raw_retirement["root_admission"]
                    == *row.admission.raw_receipt(),
            "raw3 writer requires original admission",
        )?;
        let mut bytes =
            serde_json::to_vec(&row.record).map_err(|error| invalid(error.to_string()))?;
        bytes.push(b'\n');
        let origin = RoventCorpusOrigin {
            record_sha256: row.record.record_sha256.clone(),
            input_sha256: row.record.input_sha256.clone(),
            origin_sha256: row.record.origin_sha256.clone(),
            split: row.record.split,
            game_id: row.record.input.game_id.clone(),
            opening_id: row.record.input.opening_id.clone(),
            line_genealogy_id: row.record.input.line_genealogy_id.clone(),
            transposition_sha256: row.record.input.transposition_sha256.clone(),
            actor_id: row.record.lineage.actor_id.clone(),
            teacher_actor_id: row.record.lineage.teacher_actor_id.clone(),
            ancestor_origin_sha256: row.record.lineage.ancestor_origin_sha256.clone(),
        };
        self.original.append_original_parts(
            &row.admission,
            &row.record.record_sha256,
            origin,
            bytes,
        )
    }
    pub fn flush(&mut self) -> Result<Option<RoventChunkPacket>, ArenaError> {
        self.original.flush()
    }
    fn index(&self) -> RoventCorpusIndexV2 {
        RoventCorpusIndexV2 {
            schema: ROVENT_INDEX_V2_DOMAIN.into(),
            manifest_sha256: self.original.index.manifest_sha256.clone(),
            record_count: self.original.index.record_count,
            chunks: self.original.index.chunks.clone(),
            origins: self.original.index.origins.clone(),
        }
    }
    pub fn finish(self) -> Result<RoventCorpusIndexV2, ArenaError> {
        check(
            self.original.buffer.is_empty(),
            "flush raw3 chunk before finishing",
        )?;
        let index = self.index();
        index.validate(&self.manifest)?;
        Ok(index)
    }
    pub fn finish_publication(self) -> Result<FinishedRoventCorpusV2, ArenaError> {
        self.finish_publication_retaining()
            .map_err(|rejected| rejected.failure)
    }
    pub(super) fn finish_publication_retaining(
        self,
    ) -> Result<FinishedRoventCorpusV2, Box<RoventWriterFinishFailureV2>> {
        let index = match self.prepare_publication() {
            Ok(index) => index,
            Err(failure) => {
                return Err(Box::new(RoventWriterFinishFailureV2 {
                    failure,
                    writer: self,
                }));
            }
        };
        Ok(FinishedRoventCorpusV2 {
            manifest: self.manifest,
            index,
            admissions: self.original.admissions,
            allocations: self.original.allocations,
        })
    }
    fn prepare_publication(&self) -> Result<RoventCorpusIndexV2, ArenaError> {
        check(
            self.original.buffer.is_empty()
                && !self.original.admissions.is_empty()
                && self.original.index.record_count > 0
                && self.original.index.record_count as usize == self.original.record_seals.len(),
            "raw3 finish requires flushed actual records and original admissions",
        )?;
        let index = self.index();
        index.validate(&self.manifest)?;
        for bytes in [
            serde_json::to_vec(&self.manifest),
            serde_json::to_vec(&index),
        ] {
            check(
                bytes.map_err(|error| invalid(error.to_string()))?.len() <= 16 * 1024 * 1024,
                "raw3 finished metadata cap",
            )?;
        }
        Ok(index)
    }
}

/// 원 writer의 actual PublishedRecord 소비 뒤에만 발급한다. 역직렬화/공개 생성자가 없다.
pub struct FinishedRoventCorpusV2 {
    manifest: RoventCorpusManifestV2,
    index: RoventCorpusIndexV2,
    admissions: BTreeMap<String, Arc<publication::RoventPublicationSourceSnapshot>>,
    // 비-Linux 완료 owner도 원 allocation 근거를 유지하며 게시만 Linux에 한정한다.
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    allocations: allocation::RoventAllocationSupplementCapture,
}
impl FinishedRoventCorpusV2 {
    pub fn manifest(&self) -> &RoventCorpusManifestV2 {
        &self.manifest
    }
    pub fn index(&self) -> &RoventCorpusIndexV2 {
        &self.index
    }
    pub fn native_admission_sha256(&self) -> impl Iterator<Item = &String> {
        self.admissions.keys()
    }
    #[cfg(target_os = "linux")]
    pub fn publish_atomic(
        &self,
        directory: &cap_std::fs::Dir,
        until: std::time::Instant,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> Result<RoventCorpusPublicationV2, ArenaError> {
        let manifest_bytes =
            serde_json::to_vec(&self.manifest).map_err(|error| invalid(error.to_string()))?;
        let index_bytes =
            serde_json::to_vec(&self.index).map_err(|error| invalid(error.to_string()))?;
        publication::publish_original_payloads(
            directory,
            until,
            cancel,
            publication::RoventPublicationMaterial {
                manifest_bytes: &manifest_bytes,
                index_bytes: &index_bytes,
                chunks: &self.index.chunks,
                admissions: &self.admissions,
            },
            |manifest, index, chunks, admissions, total_payload_bytes| {
                let publication = RoventCorpusPublicationV2 {
                    schema: ROVENT_CORPUS_PUBLICATION_V2.into(),
                    manifest,
                    index,
                    chunks,
                    canonical_manifest_sha256: self.manifest.sha256()?,
                    canonical_index_sha256: self.index.sha256()?,
                    source_commit: self.manifest.source_commit.clone(),
                    source_sha256: self.manifest.source_sha256.clone(),
                    runtime_sha256: self.manifest.runtime_sha256.clone(),
                    record_count: self.index.record_count,
                    total_payload_bytes,
                    native_admission_sha256: self.admissions.keys().cloned().collect(),
                    admissions,
                };
                publication.validate()?;
                Ok(publication)
            },
        )
    }
    #[cfg(target_os = "linux")]
    pub fn publish_allocation_supplement(
        &self,
        directory: &cap_std::fs::Dir,
        until: std::time::Instant,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> Result<RoventAllocationSupplement, ArenaError> {
        self.allocations
            .publish_submitted(directory, &self.manifest, &self.index, until, cancel)
    }
}
