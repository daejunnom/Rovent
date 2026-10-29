//! Bounded writer for original input-lock /2 published rows.
//!
//! It accepts only opaque in-process rows, retaining their original source
//! snapshots. Decoded records alone cannot create a writer admission.
use super::*;
use publication::PublicationSourceBytes;
use publication_lock_v2::{LockV2PublicationSource, publish_lock_v2_payloads};
use rz_experiments::rovent_corpus_lock_v2::record::{
    ROVENT_CORPUS_INDEX_LOCK_V2_DOMAIN, RoventCorpusIndexLockV2,
    RoventCorpusManifestLockV2,
};
use rz_experiments::rovent_corpus_lock_v2::publication::RoventLockV2Publication;
use std::sync::Arc;

/// Owns bounded JSONL buffers and the source snapshots of the real rows it
/// has consumed. The caller still owns chunk persistence and quota enforcement.
pub struct RoventCorpusWriterLockV2 {
    manifest: RoventCorpusManifestLockV2,
    lock: rz_experiments::RoventInputLockV2,
    index: RoventCorpusIndexLockV2,
    buffer: Vec<u8>,
    buffer_rows: u64,
    total_bytes: u64,
    output_bytes_max: u64,
    record_seals: BTreeSet<String>,
    admissions: BTreeMap<String, Arc<LockV2PublicationSource>>,
    // While accepting rows, compare the actual issuer/root owners as well as
    // their metadata. These clones retain source/Rules/root custody until
    // finish; they do not claim that the native worker is still alive.
    original_admissions: BTreeMap<String, rz_uci::pals_adaptive::RoventCollectionRootAdmissionV2>,
}

impl RoventCorpusWriterLockV2 {
    /// The manifest is checked against the original admitted lock. An
    /// admission is not a physical-completion or a teacher-result proof.
    pub fn new(
        manifest: &RoventCorpusManifestLockV2,
        admission: &rz_uci::pals_adaptive::RoventCollectionRootAdmissionV2,
    ) -> Result<Self, ArenaError> {
        let lock = admission.native_origin().original_input_lock().clone();
        manifest.validate(&lock)?;
        let manifest_bytes = serde_json::to_vec(manifest)
            .map_err(|error| invalid(error.to_string()))?
            .len() as u64;
        let output_bytes_max = lock.manifest.output_bytes_max.min(MAX_CORPUS_BYTES);
        let total_bytes = manifest_bytes.checked_add(4096)
            .ok_or_else(|| invalid("writer /2 manifest byte count overflow"))?;
        check(total_bytes <= output_bytes_max, "writer /2 initial output cap")?;
        Ok(Self {
            manifest: manifest.clone(),
            index: RoventCorpusIndexLockV2 {
                schema: ROVENT_CORPUS_INDEX_LOCK_V2_DOMAIN.into(),
                manifest_sha256: manifest.sha256(&lock)?,
                record_count: 0,
                chunks: Vec::new(),
                origins: Vec::new(),
            },
            lock,
            buffer: Vec::new(),
            buffer_rows: 0,
            total_bytes,
            output_bytes_max,
            record_seals: BTreeSet::new(),
            admissions: BTreeMap::new(),
            original_admissions: BTreeMap::new(),
        })
    }

    /// A failure leaves the original row in the caller's custody and the
    /// writer unmodified. This accepts no deserialized row or fabricated seal.
    pub fn append(&mut self, row: &PublishedRoventLockV2Record) -> Result<(), ArenaError> {
        let admission = row.admission();
        let lock = admission.native_origin().original_input_lock();
        check(lock == &self.lock, "writer /2 input lock differs")?;
        check(row.manifest_sha256() == self.index.manifest_sha256, "writer /2 manifest differs")?;
        let record = row.record();
        record.validate(&self.manifest, lock)?;
        let value = &record.0;
        check(
            value.native.raw_retirement["root_admission_sha256"] == admission.raw_sha256()
                && value.native.raw_retirement["root_admission"] == *admission.raw_receipt(),
            "writer /2 requires original root admission",
        )?;
        let mut bytes = serde_json::to_vec(record).map_err(|error| invalid(error.to_string()))?;
        bytes.push(b'\n');
        let origin = RoventCorpusOrigin {
            record_sha256: value.record_sha256.clone(),
            input_sha256: value.input_sha256.clone(),
            origin_sha256: value.origin_sha256.clone(),
            split: value.split,
            game_id: value.input.game_id.clone(),
            opening_id: value.input.opening_id.clone(),
            line_genealogy_id: value.input.line_genealogy_id.clone(),
            transposition_sha256: value.input.transposition_sha256.clone(),
            actor_id: value.lineage.actor_id.clone(),
            teacher_actor_id: value.lineage.teacher_actor_id.clone(),
            ancestor_origin_sha256: value.lineage.ancestor_origin_sha256.clone(),
        };
        let admission_sha = admission.raw_sha256().to_owned();
        if let Some(original) = self.original_admissions.get(&admission_sha) {
            check(original.same_original_admission(admission), "writer /2 original admission owner changed")?;
        }
        let (snapshot, source_bytes) = if let Some(original) = self.admissions.get(&admission_sha) {
            check(self.original_admissions.contains_key(&admission_sha), "writer /2 admission custody missing")?;
            (Arc::clone(original), 0)
        } else {
            let snapshot = Arc::new(LockV2PublicationSource::capture(admission)?);
            let bundle_bytes = snapshot.runtime_bundle().map_or(0, |(_, raw)| raw.len() as u64);
            let source_bytes = (snapshot.lock_bytes().len() as u64)
                .checked_add(bundle_bytes)
                .and_then(|n| n.checked_add(1024))
                .ok_or_else(|| invalid("writer /2 source byte count overflow"))?;
            (snapshot, source_bytes)
        };
        let origin_bytes = serde_json::to_vec(&origin)
            .map_err(|error| invalid(error.to_string()))?
            .len() as u64 + 1;
        let added = (bytes.len() as u64)
            .checked_add(origin_bytes)
            .and_then(|n| n.checked_add(source_bytes))
            .ok_or_else(|| invalid("writer /2 byte count overflow"))?;
        check(
            self.index.origins.len() < MAX_CORPUS_ROWS
                && bytes.len() <= GENERATION_MAX_ROW_BYTES + 1
                && self.buffer.len().checked_add(bytes.len()).is_some_and(|n| n <= MAX_CHUNK_BYTES)
                && self.total_bytes.checked_add(added).is_some_and(|n| n <= self.output_bytes_max)
                && !self.record_seals.contains(&value.record_sha256),
            "writer /2 output cap or duplicate; flush explicitly",
        )?;
        self.record_seals.insert(value.record_sha256.clone());
        self.admissions.entry(admission_sha).or_insert(snapshot);
        self.original_admissions.entry(admission.raw_sha256().to_owned())
            .or_insert_with(|| admission.clone());
        self.index.origins.push(origin);
        self.index.record_count += 1;
        self.buffer_rows += 1;
        self.total_bytes += added;
        self.buffer.extend(bytes);
        Ok(())
    }

    pub fn flush(&mut self) -> Result<Option<RoventChunkPacket>, ArenaError> {
        if self.buffer.is_empty() { return Ok(None); }
        check(self.index.chunks.len() < 8192, "writer /2 chunk count cap")?;
        let total = self.total_bytes.checked_add(512).ok_or_else(|| invalid("writer /2 byte count overflow"))?;
        check(total <= self.output_bytes_max, "writer /2 chunk/index cap")?;
        let bytes = std::mem::take(&mut self.buffer);
        let declaration = RoventCorpusChunk {
            path: format!("chunks/{:06}.jsonl", self.index.chunks.len()),
            bytes: bytes.len() as u64,
            sha256: hex(Digest(Sha256::digest(&bytes).into())),
            records: self.buffer_rows,
        };
        self.total_bytes = total;
        self.buffer_rows = 0;
        self.index.chunks.push(declaration.clone());
        Ok(Some(RoventChunkPacket { declaration, bytes }))
    }

    /// Finalize only the accumulated row/index assembly. This is not proof
    /// that the collection, actor fence, or teacher work completed; the
    /// returned value is partial evidence until the original lifecycle owner
    /// separately supplies a completion capability. Retain the writer on
    /// failure so its original source snapshots and pending data remain
    /// available to the caller.
    pub fn finish(self) -> Result<FinishedRoventCorpusLockV2, Box<(ArenaError, Self)>> {
        let result = (|| {
            check(self.buffer.is_empty() && self.index.record_count > 0
                && !self.admissions.is_empty()
                && self.index.record_count as usize == self.record_seals.len(),
                "writer /2 finish requires flushed original rows")?;
            self.index.validate(&self.manifest, &self.lock)?;
            Ok::<(), ArenaError>(())
        })();
        if let Err(error) = result { return Err(Box::new((error, self))); }
        Ok(FinishedRoventCorpusLockV2 { manifest: self.manifest, index: self.index, admissions: self.admissions })
    }
}

/// Issued only by the writer after consuming original opaque rows. This is a
/// partial assembled corpus, not evidence of whole-run completion; it cannot
/// restore live admission, acceptance, or physical-completion authority.
pub struct FinishedRoventCorpusLockV2 {
    manifest: RoventCorpusManifestLockV2,
    index: RoventCorpusIndexLockV2,
    admissions: BTreeMap<String, Arc<LockV2PublicationSource>>,
}

impl FinishedRoventCorpusLockV2 {
    pub fn manifest(&self) -> &RoventCorpusManifestLockV2 { &self.manifest }
    pub fn index(&self) -> &RoventCorpusIndexLockV2 { &self.index }

    // The original session owner has not yet supplied an unforgeable actor
    // fence and teacher-terminal completion capability. Keep this internal
    // until the lifecycle owner consumes and checks that capability; callers
    // can still retain the writer, chunks, and individual retired rows.
    #[allow(dead_code)]
    pub(crate) fn publish_atomic(
        &self,
        directory: &cap_std::fs::Dir,
        until: std::time::Instant,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> Result<RoventLockV2Publication, ArenaError> {
        publish_lock_v2_payloads(directory, until, cancel, &self.manifest, &self.index, &self.admissions)
    }
}
