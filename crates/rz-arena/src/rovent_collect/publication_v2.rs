//! submitted-input/1 + raw/3 자료의 별도 publication 형식이다.
//! 기존 publication/1을 parser에서 재서명하거나 넓혀 받지 않는다.
use super::*;
use serde::{Deserialize, Serialize};

pub const ROVENT_CORPUS_PUBLICATION_V2: &str = "rz-rovent-corpus-publication/2";

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventCorpusPublicationV2 {
    pub schema: String,
    pub manifest: RoventPublicationFile,
    pub index: RoventPublicationFile,
    pub chunks: Vec<RoventPublicationFile>,
    pub canonical_manifest_sha256: String,
    pub canonical_index_sha256: String,
    pub source_commit: String,
    pub source_sha256: String,
    pub runtime_sha256: String,
    pub record_count: u64,
    pub total_payload_bytes: u64,
    pub native_admission_sha256: Vec<String>,
    pub admissions: Vec<RoventPublicationAdmission>,
}
impl RoventCorpusPublicationV2 {
    /// 파일 pin과 allocation bounds의 기존 순수 검사를 공유한다. projection은
    /// 원 /1 문서·writer 권한·모델/Rules 검증으로 발행하지 않는다.
    pub fn validate(&self) -> Result<(), ArenaError> {
        check(
            self.schema == ROVENT_CORPUS_PUBLICATION_V2,
            "explicit publication/2 required",
        )?;
        RoventCorpusPublication {
            schema: publication::ROVENT_CORPUS_PUBLICATION.into(),
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
            admissions: self.admissions.clone(),
        }
        .validate()
    }
}
