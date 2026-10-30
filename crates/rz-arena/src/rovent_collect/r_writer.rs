//! 원 R 퇴역 전체의 streaming publication이다. 큰 source bytes는 즉시
//! 원 private-directory sink에 넘기고 writer에는 작은 pin만 남긴다.
//! 실패 시 poison하며 caller의 원 capture와 부분 파일은 보존한다.
use super::{r_publication::*, r_readout::CapturedRoventRRetirement, *};
use rz_experiments::rovent_r_readout::*;
use rz_uci::rovent_bootstrap::RoventStartupControl;
use std::sync::Arc;

pub struct RoventRPublicationPacket {
    pub declaration: RoventPublicationFile,
    pub bytes: Vec<u8>,
}
type Payloads = BTreeMap<String, RoventPublicationFile>;
type Sink<'a> = dyn FnMut(RoventRPublicationPacket) -> Result<(), ArenaError> + 'a;
fn emit(
    path: String,
    bytes: Vec<u8>,
    pins: &mut Payloads,
    sink: &mut Sink<'_>,
) -> Result<RoventPublicationFile, ArenaError> {
    let pin = RoventPublicationFile {
        path: path.clone(),
        sha256: hex(Digest(Sha256::digest(&bytes).into())),
        bytes: bytes.len() as u64,
    };
    pin.validate()?;
    if let Some(old) = pins.get(&path) {
        check(old == &pin, "R payload path conflict")?;
        return Ok(pin);
    }
    let total = pins
        .values()
        .try_fold(pin.bytes, |n, p| n.checked_add(p.bytes));
    check(
        total.is_some_and(|n| n <= MAX_CORPUS_BYTES),
        "R payload 18GiB cap",
    )?;
    sink(RoventRPublicationPacket {
        declaration: pin.clone(),
        bytes,
    })?;
    pins.insert(path, pin.clone());
    Ok(pin)
}
struct SourcePins {
    lock: RoventPublicationFile,
    canonical_lock: String,
    bundle: Option<RoventPublicationFile>,
    weights: RoventPublicationFile,
    provenance: RoventPublicationFile,
    parameter: String,
    registration: Value,
}
impl SourcePins {
    fn capture(
        captured: &CapturedRoventRRetirement,
        control: &RoventStartupControl,
        pins: &mut Payloads,
        sink: &mut Sink<'_>,
    ) -> Result<Self, ArenaError> {
        control.check().map_err(|e| invalid(e.to_string()))?;
        let origin = captured.admission().native_origin();
        let lock = origin.original_input_lock();
        lock.validate()?;
        let source = captured.nnuec_source();
        check(
            lock.canonical_sha256 == hex(source.input_lock_digest())
                && source.input_lock_digest()
                    == captured.admission().root_admission().startup_lock_digest(),
            "R original source lock",
        )?;
        let raw = serde_json::to_vec(lock).map_err(|e| invalid(e.to_string()))?;
        check(raw.len() <= 16 * 1024 * 1024, "R lock metadata cap")?;
        let lock_pin = emit(
            format!("input-locks/{}.json", lock.canonical_sha256),
            raw,
            pins,
            sink,
        )?;
        let bundle = origin
            .runtime_bundle_publication_bytes()
            .map_err(|e| invalid(e.to_string()))?
            .map(|(original, raw)| {
                let pin = emit(
                    format!("runtime-bundles/{}.json", original.sha256),
                    raw,
                    pins,
                    sink,
                )?;
                check(
                    pin.sha256 == original.sha256 && pin.bytes == original.bytes,
                    "R original bundle raw bytes",
                )?;
                Ok::<_, ArenaError>(pin)
            })
            .transpose()?;
        let (weights, raw_weights, provenance, raw_provenance) = source
            .publication_source_bytes(control)
            .map_err(|e| invalid(e.to_string()))?;
        let weights_pin = emit(
            format!("nnuec-weights/{}.json", weights.sha256),
            raw_weights,
            pins,
            sink,
        )?;
        let provenance_pin = emit(
            format!("nnuec-provenance/{}.json", provenance.sha256),
            raw_provenance,
            pins,
            sink,
        )?;
        check(
            weights_pin.sha256 == weights.sha256
                && weights_pin.bytes == weights.bytes
                && provenance_pin.sha256 == provenance.sha256
                && provenance_pin.bytes == provenance.bytes,
            "R original NNUEC raw file bytes",
        )?;
        control.check().map_err(|e| invalid(e.to_string()))?;
        Ok(Self {
            lock: lock_pin,
            canonical_lock: lock.canonical_sha256.clone(),
            bundle,
            weights: weights_pin,
            provenance: provenance_pin,
            parameter: source
                .semantic_weights_identity()
                .weights_sha256
                .clone()
                .ok_or_else(|| invalid("R semantic parameters absent"))?,
            registration: serde_json::to_value(source.registration())
                .map_err(|e| invalid(e.to_string()))?,
        })
    }
    fn admission(&self, sha: String) -> Result<RoventRPublicationAdmission, ArenaError> {
        Ok(RoventRPublicationAdmission {
            native_admission_sha256: sha,
            input_lock: self.lock.clone(),
            canonical_input_lock_sha256: self.canonical_lock.clone(),
            runtime_bundle_manifest: self.bundle.clone(),
            nnuec_weights: self.weights.clone(),
            nnuec_provenance: self.provenance.clone(),
            nnuec_parameter_sha256: self.parameter.clone(),
            nnuec_registration: self.registration.clone(),
            nnuec_registration_sha256: generation_digest(
                "rz-rovent-r-nnuec-registration/1",
                &self.registration,
            )?,
        })
    }
}
pub struct RoventRCorpusWriter {
    manifest: RoventRReadoutManifest,
    index: RoventRReadoutIndex,
    buffer: Vec<u8>,
    buffer_rows: u64,
    metadata_bytes: u64,
    records: BTreeSet<String>,
    payloads: Payloads,
    admissions: BTreeMap<String, Arc<SourcePins>>,
    sources: BTreeMap<(String, String, String), Arc<SourcePins>>,
    retirements: BTreeMap<String, RoventRPublicationRetirement>,
    poisoned: bool,
    allocations: super::allocation::RoventAllocationSupplementCapture,
}
impl RoventRCorpusWriter {
    pub fn new(manifest: &RoventRReadoutManifest) -> Result<Self, ArenaError> {
        let bytes = serde_json::to_vec(manifest)
            .map_err(|e| invalid(e.to_string()))?
            .len();
        check(bytes <= 16 * 1024 * 1024, "R manifest metadata cap")?;
        Ok(Self {
            manifest: manifest.clone(),
            index: RoventRReadoutIndex {
                schema: R_READOUT_INDEX_DOMAIN.into(),
                manifest_sha256: manifest.sha256()?,
                record_count: 0,
                chunks: vec![],
                origins: vec![],
            },
            buffer: vec![],
            buffer_rows: 0,
            metadata_bytes: bytes as u64 + 4096,
            records: BTreeSet::new(),
            payloads: BTreeMap::new(),
            admissions: BTreeMap::new(),
            sources: BTreeMap::new(),
            retirements: BTreeMap::new(),
            poisoned: false,
            allocations: super::allocation::RoventAllocationSupplementCapture::new(),
        })
    }
    fn total(&self) -> Result<(), ArenaError> {
        let total = self
            .payloads
            .values()
            .try_fold(self.metadata_bytes, |n, p| n.checked_add(p.bytes))
            .and_then(|n| n.checked_add(self.buffer.len() as u64));
        check(
            total.is_some_and(|n| n <= MAX_CORPUS_BYTES),
            "R aggregate corpus 18GiB cap",
        )
    }
    fn charge(&mut self, n: u64) -> Result<(), ArenaError> {
        self.metadata_bytes = self
            .metadata_bytes
            .checked_add(n)
            .ok_or_else(|| invalid("R metadata overflow"))?;
        self.total()
    }
    fn flush(&mut self, sink: &mut Sink<'_>) -> Result<(), ArenaError> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        check(self.index.chunks.len() < 8192, "R chunk count cap")?;
        self.charge(512)?;
        let bytes = std::mem::take(&mut self.buffer);
        let pin = emit(
            format!("r-chunks/{:06}.jsonl", self.index.chunks.len()),
            bytes,
            &mut self.payloads,
            sink,
        )?;
        self.index.chunks.push(RoventCorpusChunk {
            path: pin.path,
            bytes: pin.bytes,
            sha256: pin.sha256,
            records: self.buffer_rows,
        });
        self.buffer_rows = 0;
        self.total()
    }
    /// 한 원 퇴역의 모든 retained readout을 함께 받는다. sink 실패 시 이 writer는
    /// poison되어 finish할 수 없으며 원 captured owner는 caller에게 남는다.
    pub fn append_retirement(
        &mut self,
        captured: &CapturedRoventRRetirement,
        control: &RoventStartupControl,
        mut sink: impl FnMut(RoventRPublicationPacket) -> Result<(), ArenaError>,
    ) -> Result<(), ArenaError> {
        check(!self.poisoned, "R writer retains partial failure")?;
        self.poisoned = true;
        self.append_inner(captured, control, &mut sink)?;
        self.poisoned = false;
        Ok(())
    }
    fn append_inner(
        &mut self,
        captured: &CapturedRoventRRetirement,
        control: &RoventStartupControl,
        sink: &mut Sink<'_>,
    ) -> Result<(), ArenaError> {
        control.check().map_err(|e| invalid(e.to_string()))?;
        let retirement = captured.retirement();
        let sha = captured.retirement_sha256();
        check(
            captured.manifest_sha256() == self.index.manifest_sha256
                && !self.retirements.contains_key(sha)
                && self.retirements.len() < MAX_CORPUS_ROWS
                && captured.rows().len() == usize::from(retirement.retained_readouts)
                && generation_digest(R_READOUT_RETIREMENT_DOMAIN, retirement)? == sha,
            "R original complete retirement manifest/count/seal",
        )?;
        let admission = captured.admission();
        let allocation_bytes = self.allocations.retain_admission(admission)?;
        self.charge(allocation_bytes)?;
        let source = captured.nnuec_source();
        let key = (
            hex(source.input_lock_digest()),
            source.weights_artifact().sha256.clone(),
            source.provenance_artifact().sha256.clone(),
        );
        let pins = if let Some(pins) = self.sources.get(&key) {
            Arc::clone(pins)
        } else {
            let pins = Arc::new(SourcePins::capture(
                captured,
                control,
                &mut self.payloads,
                sink,
            )?);
            self.total()?;
            self.sources.insert(key, Arc::clone(&pins));
            pins
        };
        let admission_sha = admission.raw_sha256().to_owned();
        if let Some(old) = self.admissions.get(&admission_sha) {
            check(Arc::ptr_eq(old, &pins), "R admission source conflict")?;
        } else {
            self.charge(2048)?;
            self.admissions.insert(admission_sha.clone(), pins);
        }
        let mut readout_shas = BTreeSet::new();
        let mut attempts = BTreeSet::new();
        for original in captured.rows() {
            control.check().map_err(|e| invalid(e.to_string()))?;
            let row = original.record();
            row.validate(&self.manifest)?;
            check(
                original.admission().raw_sha256() == admission_sha
                    && Arc::ptr_eq(original.nnuec_source(), source)
                    && row.native_admission_sha256 == admission_sha
                    && row.native_admission == *admission.raw_receipt()
                    && row.retirement == *retirement
                    && row.retirement_sha256 == sha
                    && !self.records.contains(&row.record_sha256)
                    && readout_shas.insert(row.record_sha256.clone())
                    && attempts.insert(row.question.readout_sequence),
                "R original row/source/retirement coverage",
            )?;
            let mut bytes = serde_json::to_vec(row).map_err(|e| invalid(e.to_string()))?;
            bytes.push(b'\n');
            check(bytes.len() <= MAX_CHUNK_BYTES, "R bounded row/chunk")?;
            if self.buffer.len() + bytes.len() > MAX_CHUNK_BYTES {
                self.flush(sink)?;
            }
            let origin = RoventRReadoutOrigin::from_record(row);
            self.charge(
                serde_json::to_vec(&origin)
                    .map_err(|e| invalid(e.to_string()))?
                    .len() as u64
                    + 1,
            )?;
            check(
                self.index.origins.len() < MAX_CORPUS_ROWS,
                "R row count cap",
            )?;
            self.records.insert(row.record_sha256.clone());
            self.index.origins.push(origin);
            self.index.record_count += 1;
            self.buffer_rows += 1;
            self.buffer.extend(bytes);
            self.total()?;
        }
        let shas = readout_shas.into_iter().collect::<Vec<_>>();
        let body = json!({"schema":ROVENT_R_RETIREMENT_OBSERVATION_DOMAIN,"native_admission":admission.raw_receipt(),"native_admission_sha256":admission_sha,
            "retirement":retirement,"retirement_sha256":sha,"readout_record_sha256":shas});
        let raw = serde_json::to_vec(&body).map_err(|e| invalid(e.to_string()))?;
        check(raw.len() <= 16 * 1024 * 1024, "R retirement metadata cap")?;
        let file = emit(
            format!("r-retirements/{sha}.json"),
            raw,
            &mut self.payloads,
            sink,
        )?;
        self.charge(512 + shas.len() as u64 * 67)?;
        self.retirements.insert(
            sha.into(),
            RoventRPublicationRetirement {
                retirement_sha256: sha.into(),
                file,
                readout_record_sha256: shas,
            },
        );
        Ok(())
    }
    pub fn finish(
        self,
        mut sink: impl FnMut(RoventRPublicationPacket) -> Result<(), ArenaError>,
    ) -> Result<FinishedRoventRCorpus, ArenaError> {
        self.finish_retaining(&mut sink)
            .map_err(|rejected| rejected.error)
    }
    /// 실패한 flush/metadata 검증에서도 같은 writer와 원 source pin을 반환한다.
    pub fn finish_retaining(
        mut self,
        mut sink: impl FnMut(RoventRPublicationPacket) -> Result<(), ArenaError>,
    ) -> Result<FinishedRoventRCorpus, Box<RoventRWriterFinishRejected>> {
        let admissions = match self.prepare_finish(&mut sink) {
            Ok(admissions) => admissions,
            Err(error) => {
                return Err(Box::new(RoventRWriterFinishRejected {
                    error,
                    writer: self,
                }));
            }
        };
        Ok(FinishedRoventRCorpus {
            manifest: self.manifest,
            index: self.index,
            payloads: self.payloads,
            admissions,
            retirements: self.retirements.into_values().collect(),
            allocations: self.allocations,
        })
    }
    fn prepare_finish(
        &mut self,
        sink: &mut Sink<'_>,
    ) -> Result<Vec<RoventRPublicationAdmission>, ArenaError> {
        check(
            !self.poisoned && !self.retirements.is_empty(),
            "R finish requires complete actual retirements",
        )?;
        self.poisoned = true;
        self.flush(sink)?;
        self.index.validate()?;
        self.admissions
            .iter()
            .map(|(sha, pins)| pins.admission(sha.clone()))
            .collect::<Result<Vec<_>, _>>()
    }
}
pub struct RoventRWriterFinishRejected {
    pub error: ArenaError,
    pub writer: RoventRCorpusWriter,
}
pub struct FinishedRoventRCorpus {
    manifest: RoventRReadoutManifest,
    index: RoventRReadoutIndex,
    payloads: Payloads,
    admissions: Vec<RoventRPublicationAdmission>,
    retirements: Vec<RoventRPublicationRetirement>,
    // 비-Linux R 완료 owner도 원 allocation 근거를 보유한다. 게시만 Linux 전용이다.
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    allocations: super::allocation::RoventAllocationSupplementCapture,
}
impl FinishedRoventRCorpus {
    pub fn manifest(&self) -> &RoventRReadoutManifest {
        &self.manifest
    }
    pub fn index(&self) -> &RoventRReadoutIndex {
        &self.index
    }
    /// Pure declaration of the original memory-sink packets. No file is
    /// written, synced or acknowledged, and no live/physical authority is
    /// minted. The eventual publisher verifies these same raw pins again.
    pub fn planned_publication(&self) -> Result<RoventRCorpusPublication, ArenaError> {
        let manifest_raw =
            serde_json::to_vec(&self.manifest).map_err(|error| invalid(error.to_string()))?;
        let index_raw =
            serde_json::to_vec(&self.index).map_err(|error| invalid(error.to_string()))?;
        check(
            manifest_raw.len() <= 16 << 20 && index_raw.len() <= 16 << 20,
            "R planned final metadata cap",
        )?;
        let pin = |path: &str, raw: &[u8]| RoventPublicationFile {
            path: path.into(),
            sha256: hex(Digest(Sha256::digest(raw).into())),
            bytes: raw.len() as u64,
        };
        let manifest = pin("r-corpus-manifest.json", &manifest_raw);
        let index = pin("r-corpus-index.json", &index_raw);
        let total = self
            .payloads
            .values()
            .try_fold(manifest.bytes + index.bytes, |sum, item| {
                sum.checked_add(item.bytes)
            })
            .ok_or_else(|| invalid("R planned payload overflow"))?;
        self.publication_declaration(manifest, index, total)
    }
    fn publication_declaration(
        &self,
        manifest: RoventPublicationFile,
        index: RoventPublicationFile,
        total_payload_bytes: u64,
    ) -> Result<RoventRCorpusPublication, ArenaError> {
        let publication = RoventRCorpusPublication {
            schema: ROVENT_R_PUBLICATION_DOMAIN.into(),
            manifest,
            index,
            chunks: self
                .index
                .chunks
                .iter()
                .map(|chunk| RoventPublicationFile {
                    path: chunk.path.clone(),
                    sha256: chunk.sha256.clone(),
                    bytes: chunk.bytes,
                })
                .collect(),
            retirements: self.retirements.clone(),
            canonical_manifest_sha256: self.manifest.sha256()?,
            canonical_index_sha256: self.index.sha256()?,
            source_commit: self.manifest.collection.source_commit.clone(),
            source_sha256: self.manifest.collection.source_sha256.clone(),
            runtime_sha256: self.manifest.collection.runtime_sha256.clone(),
            record_count: self.index.record_count,
            total_payload_bytes,
            admissions: self.admissions.clone(),
        };
        publication.validate()?;
        Ok(publication)
    }
    #[cfg(target_os = "linux")]
    pub fn publish_allocation_supplement(
        &self,
        directory: &cap_std::fs::Dir,
        until: std::time::Instant,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> Result<super::allocation::RoventAllocationSupplement, ArenaError> {
        self.allocations
            .publish_r(directory, &self.manifest, &self.index, until, cancel)
    }
    /// 원 private directory의 모든 packet을 재해시·sync한 뒤 no-overwrite commit한다.
    #[cfg(target_os = "linux")]
    pub fn publish_atomic(
        &self,
        directory: &cap_std::fs::Dir,
        until: std::time::Instant,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> Result<RoventRCorpusPublication, ArenaError> {
        use cap_fs_ext::{FollowSymlinks, OpenOptionsFollowExt};
        use std::io::{Read, Write};
        let io = |e: std::io::Error| ArenaError::Io(e.to_string());
        let active = || {
            check(
                !cancel.load(std::sync::atomic::Ordering::Acquire)
                    && std::time::Instant::now() < until,
                "R publication canceled/expired; partial files preserved",
            )
        };
        let put = |path: &str, raw: &[u8]| -> Result<RoventPublicationFile, ArenaError> {
            active()?;
            let pin = RoventPublicationFile {
                path: path.into(),
                sha256: hex(Digest(Sha256::digest(raw).into())),
                bytes: raw.len() as u64,
            };
            pin.validate()?;
            let mut opts = cap_std::fs::OpenOptions::new();
            opts.write(true).create_new(true).follow(FollowSymlinks::No);
            let mut file = directory.open_with(path, &opts).map_err(io)?;
            for part in raw.chunks(64 * 1024) {
                active()?;
                file.write_all(part).map_err(io)?;
            }
            file.sync_all().map_err(io)?;
            check(file.metadata().map_err(io)?.is_file(), "R regular metadata")?;
            Ok(pin)
        };
        let mut all = self.payloads.clone();
        for pin in self.payloads.values() {
            active()?;
            pin.validate()?;
            let mut opts = cap_std::fs::OpenOptions::new();
            opts.read(true).follow(FollowSymlinks::No);
            let mut file = directory.open_with(&pin.path, &opts).map_err(io)?;
            let meta = file.metadata().map_err(io)?;
            check(
                meta.is_file() && meta.len() == pin.bytes,
                "R packet original type/bytes",
            )?;
            let mut reader = Read::take(&mut file, pin.bytes + 1);
            let mut buffer = [0u8; 64 * 1024];
            let mut count = 0u64;
            let mut hash = Sha256::new();
            loop {
                active()?;
                let n = reader.read(&mut buffer).map_err(io)?;
                if n == 0 {
                    break;
                }
                count += n as u64;
                hash.update(&buffer[..n]);
            }
            check(
                count == pin.bytes && hex(Digest(hash.finalize().into())) == pin.sha256,
                "R packet raw bytes changed",
            )?;
            file.sync_all().map_err(io)?;
        }
        let manifest_raw =
            serde_json::to_vec(&self.manifest).map_err(|e| invalid(e.to_string()))?;
        let index_raw = serde_json::to_vec(&self.index).map_err(|e| invalid(e.to_string()))?;
        check(
            manifest_raw.len() <= 16 * 1024 * 1024 && index_raw.len() <= 16 * 1024 * 1024,
            "R final metadata cap",
        )?;
        let manifest = put("r-corpus-manifest.json", &manifest_raw)?;
        all.insert(manifest.path.clone(), manifest.clone());
        let index = put("r-corpus-index.json", &index_raw)?;
        all.insert(index.path.clone(), index.clone());
        let total = all
            .values()
            .try_fold(0u64, |n, p| n.checked_add(p.bytes))
            .ok_or_else(|| invalid("R payload overflow"))?;
        let publication = self.publication_declaration(manifest, index, total)?;
        let raw = serde_json::to_vec(&publication).map_err(|e| invalid(e.to_string()))?;
        check(raw.len() <= 16 * 1024 * 1024, "R final commit cap")?;
        put("r-publication.pending.json", &raw)?;
        for name in self
            .payloads
            .keys()
            .filter_map(|p| p.split_once('/').map(|v| v.0))
            .collect::<BTreeSet<_>>()
        {
            let sub = directory.open_dir(name).map_err(io)?;
            crate::pals_collect_v3::training_process::directory_file(&sub)?
                .sync_all()
                .map_err(io)?;
        }
        let root = crate::pals_collect_v3::training_process::directory_file(directory)?;
        root.sync_all().map_err(io)?;
        active()?;
        directory
            .hard_link(
                "r-publication.pending.json",
                directory,
                "r-publication.json",
            )
            .map_err(io)?;
        directory
            .remove_file("r-publication.pending.json")
            .map_err(io)?;
        root.sync_all().map_err(io)?;
        Ok(publication)
    }
}
