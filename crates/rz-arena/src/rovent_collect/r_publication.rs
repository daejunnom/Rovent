//! 별도 R 원자료 publication의 미등록 wire 초안이다.
//! 이 DTO의 무결성 확인은 원 capture/semantic/학습 권한을 복원하지 않는다.
//! 원 R writer만 CapturedRoventRRetirement 전부를 보존한 뒤 commit을 게시한다.
use super::*;
use serde::{Deserialize, Serialize};

pub const ROVENT_R_PUBLICATION_DOMAIN: &str = "rz-rovent-r-readout-publication/1";
pub const ROVENT_R_RETIREMENT_OBSERVATION_DOMAIN: &str = "rz-rovent-r-retirement-observation/1";

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventRPublicationAdmission {
    pub native_admission_sha256: String,
    pub input_lock: RoventPublicationFile,
    pub canonical_input_lock_sha256: String,
    pub runtime_bundle_manifest: Option<RoventPublicationFile>,
    /// Both are original file bytes, not numerical parameter digests.
    pub nnuec_weights: RoventPublicationFile,
    pub nnuec_provenance: RoventPublicationFile,
    pub nnuec_parameter_sha256: String,
    pub nnuec_registration: Value,
    pub nnuec_registration_sha256: String,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventRPublicationRetirement {
    /// Whole original capture retirement; all of its retained readouts are
    /// present in this corpus even if they cross individual chunk boundaries.
    pub retirement_sha256: String,
    pub file: RoventPublicationFile,
    pub readout_record_sha256: Vec<String>,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventRCorpusPublication {
    pub schema: String,
    pub manifest: RoventPublicationFile,
    pub index: RoventPublicationFile,
    pub chunks: Vec<RoventPublicationFile>,
    pub retirements: Vec<RoventRPublicationRetirement>,
    pub canonical_manifest_sha256: String,
    pub canonical_index_sha256: String,
    pub source_commit: String,
    pub source_sha256: String,
    pub runtime_sha256: String,
    pub record_count: u64,
    /// Unique payload paths, excluding this final commit file itself.
    pub total_payload_bytes: u64,
    pub admissions: Vec<RoventRPublicationAdmission>,
}
impl RoventRCorpusPublication {
    pub fn validate(&self) -> Result<(), ArenaError> {
        check(
            self.schema == ROVENT_R_PUBLICATION_DOMAIN
                && self.manifest.path == "r-corpus-manifest.json"
                && self.index.path == "r-corpus-index.json"
                && self.source_commit.len() == 40
                && self
                    .source_commit
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                && [
                    &self.canonical_manifest_sha256,
                    &self.canonical_index_sha256,
                    &self.source_sha256,
                    &self.runtime_sha256,
                ]
                .iter()
                .all(|sha| generation_sha(sha))
                && self.record_count <= MAX_CORPUS_ROWS as u64
                && self.chunks.len() <= 8192
                && !self.retirements.is_empty()
                && self.retirements.len() <= MAX_CORPUS_ROWS
                && !self.admissions.is_empty()
                && self.admissions.len() <= MAX_CORPUS_ROWS
                && self
                    .admissions
                    .windows(2)
                    .all(|v| v[0].native_admission_sha256 < v[1].native_admission_sha256),
            "R publication domain/source/count/admission ordering",
        )?;
        let mut files: BTreeMap<String, RoventPublicationFile> = BTreeMap::new();
        let mut add = |pin: &RoventPublicationFile| -> Result<(), ArenaError> {
            pin.validate()?;
            if let Some(original) = files.get(pin.path.as_str()) {
                check(
                    original == pin,
                    "R publication path has conflicting file pins",
                )?;
            } else {
                files.insert(pin.path.clone(), pin.clone());
            }
            Ok(())
        };
        add(&self.manifest)?;
        add(&self.index)?;
        let mut chunk_paths = BTreeSet::new();
        for chunk in &self.chunks {
            check(
                chunk.path.starts_with("r-chunks/")
                    && chunk.path.ends_with(".jsonl")
                    && chunk.bytes <= MAX_CHUNK_BYTES as u64
                    && chunk_paths.insert(&chunk.path),
                "R publication chunk path/cap/duplicate",
            )?;
            add(chunk)?;
        }
        let mut retired = BTreeSet::new();
        let mut rows = BTreeSet::new();
        for terminal in &self.retirements {
            check(
                generation_sha(&terminal.retirement_sha256)
                    && terminal.file.path
                        == format!("r-retirements/{}.json", terminal.retirement_sha256)
                    && terminal.file.bytes <= 16 * 1024 * 1024
                    && terminal.readout_record_sha256.len() <= 64
                    && terminal
                        .readout_record_sha256
                        .iter()
                        .all(|sha| generation_sha(sha))
                    && terminal
                        .readout_record_sha256
                        .windows(2)
                        .all(|w| w[0] < w[1])
                    && terminal
                        .readout_record_sha256
                        .iter()
                        .all(|sha| rows.insert(sha))
                    && retired.insert(&terminal.retirement_sha256),
                "R complete retirement/readout coverage",
            )?;
            add(&terminal.file)?;
        }
        check(
            rows.len() as u64 == self.record_count,
            "R publication every record belongs to one actual retirement",
        )?;
        let mut source_files: BTreeMap<(&str, &str), &RoventPublicationFile> = BTreeMap::new();
        for origin in &self.admissions {
            check(
                generation_sha(&origin.native_admission_sha256)
                    && generation_sha(&origin.canonical_input_lock_sha256)
                    && generation_sha(&origin.nnuec_parameter_sha256)
                    && generation_sha(&origin.nnuec_registration_sha256)
                    && origin.input_lock.path
                        == format!("input-locks/{}.json", origin.canonical_input_lock_sha256)
                    && origin.nnuec_weights.path
                        == format!("nnuec-weights/{}.json", origin.nnuec_weights.sha256)
                    && origin.nnuec_weights.bytes <= 32 * 1024 * 1024
                    && origin.nnuec_provenance.path
                        == format!("nnuec-provenance/{}.json", origin.nnuec_provenance.sha256)
                    && origin.nnuec_provenance.bytes <= 1024 * 1024,
                "R publication original input lock/NNUEC source pins",
            )?;
            // Registration has only strings/integers and the original typed
            // model declaration. This seal is not a live model/source issuer.
            check(
                generation_digest(
                    "rz-rovent-r-nnuec-registration/1",
                    &origin.nnuec_registration,
                )? == origin.nnuec_registration_sha256,
                "R original NNUEC registration projection",
            )?;
            for (kind, pin) in [
                ("lock", &origin.input_lock),
                ("weights", &origin.nnuec_weights),
                ("provenance", &origin.nnuec_provenance),
            ] {
                if let Some(original) = source_files.get(&(kind, pin.sha256.as_str())) {
                    check(
                        *original == pin,
                        "one R source raw SHA has conflicting file pins",
                    )?;
                } else {
                    source_files.insert((kind, &pin.sha256), pin);
                }
                add(pin)?;
            }
            if let Some(bundle) = &origin.runtime_bundle_manifest {
                check(
                    bundle.path == format!("runtime-bundles/{}.json", bundle.sha256)
                        && bundle.bytes <= 64 * 1024,
                    "R original bundle file pin; distinct from its spec digest",
                )?;
                add(bundle)?;
            }
        }
        let bytes = files
            .values()
            .try_fold(0u64, |n, pin| n.checked_add(pin.bytes))
            .ok_or_else(|| invalid("R publication payload byte overflow"))?;
        check(
            bytes == self.total_payload_bytes && bytes <= MAX_CORPUS_BYTES,
            "R exact unique-path payload bytes",
        )
    }
}
