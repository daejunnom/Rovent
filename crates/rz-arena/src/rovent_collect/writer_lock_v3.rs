//! Bounded writer for opaque records from the original input-lock /3 owner.
//!
//! Rows cannot be constructed from decoded bytes or a copied digest. The
//! writer retains the actual root admissions while it accepts records; its
//! finished value is assembled evidence, not a physical-completion proof.
use super::*;
use publication::PublicationSourceBytes;
use publication_lock_v3::{LockV3PublicationSource, publish_lock_v3_payloads};
use rz_experiments::rovent_corpus_lock_v3::publication::RoventLockV3Publication;
use rz_experiments::rovent_corpus_lock_v3::record::{
    ROVENT_CORPUS_INDEX_LOCK_V3_DOMAIN, RoventCorpusIndexLockV3, RoventCorpusManifestLockV3,
};
use std::sync::Arc;

/// Bounded JSONL assembly with the original /3 root owners held until finish.
pub struct RoventCorpusWriterLockV3 {
    manifest: RoventCorpusManifestLockV3,
    lock: rz_experiments::RoventInputLockV3,
    index: RoventCorpusIndexLockV3,
    buffer: Vec<u8>,
    buffer_rows: u64,
    total_bytes: u64,
    output_bytes_max: u64,
    record_seals: BTreeSet<String>,
    admissions: BTreeMap<String, Arc<LockV3PublicationSource>>,
    original_admissions: BTreeMap<String, rz_uci::pals_adaptive::RoventCollectionRootAdmissionV3>,
    // A caught panic during gated append may have changed part of the writer.
    // Keep the original payload and prohibit flush, retry, or finish.
    uncertain_append: Option<Box<dyn std::any::Any + Send>>,
}

/// The caller always retains the original pending row. Busy is retryable;
/// Uncertain retains both original writer and policy owner for recovery.
pub(super) enum PolicyAppendResultV3 {
    Appended,
    Busy,
    Rejected(ArenaError),
    GateRejected(super::speculative_route_v3::TargetBeginRejectionV3),
    /// Candidate cloning panicked before the gate or writer was touched.
    PreparationPanicked(Box<dyn std::any::Any + Send>),
    Uncertain,
}

impl RoventCorpusWriterLockV3 {
    /// Failure-only access to the bytes already appended; this issues no
    /// publication authority and leaves the original owner unchanged.
    pub(crate) fn failure_buffer(&self) -> (&[u8], u64) {
        (&self.buffer, self.buffer_rows)
    }

    /// Whether this exact row requires flushing before it can be appended.
    /// All ownership, validation, duplicate and output-budget checks still
    /// take place in `append`.
    pub(crate) fn chunk_flush_required(
        &self,
        row: &PublishedRoventLockV3Record,
    ) -> Result<bool, ArenaError> {
        check(
            self.uncertain_append.is_none(),
            "writer /3 append state uncertain",
        )?;
        let row_bytes = serde_json::to_vec(row.record())
            .map_err(|error| invalid(error.to_string()))?
            .len()
            .checked_add(1)
            .ok_or_else(|| invalid("writer /3 row byte count overflow"))?;
        Ok(!self.buffer.is_empty()
            && row_bytes <= MAX_CHUNK_BYTES
            && self
                .buffer
                .len()
                .checked_add(row_bytes)
                .is_some_and(|n| n > MAX_CHUNK_BYTES))
    }

    /// Size check only; the opaque candidate and original target capability
    /// stay in the caller's slot and no publication authority is exercised.
    pub(super) fn policy_chunk_flush_required(
        &self,
        pending: &super::producer_v3::PendingPolicyPublicationV3,
    ) -> Result<bool, ArenaError> {
        check(
            self.uncertain_append.is_none(),
            "writer /3 append state uncertain",
        )?;
        let row_bytes = serde_json::to_vec(pending.candidate())
            .map_err(|error| invalid(error.to_string()))?
            .len()
            .checked_add(1)
            .ok_or_else(|| invalid("writer /3 row byte count overflow"))?;
        Ok(!self.buffer.is_empty()
            && row_bytes <= MAX_CHUNK_BYTES
            && self
                .buffer
                .len()
                .checked_add(row_bytes)
                .is_some_and(|n| n > MAX_CHUNK_BYTES))
    }

    /// Validate against the admitted /3 lock; this is not evidence that the
    /// worker, accelerator, teacher or full collection has completed.
    pub fn new(
        manifest: &RoventCorpusManifestLockV3,
        admission: &rz_uci::pals_adaptive::RoventCollectionRootAdmissionV3,
    ) -> Result<Self, ArenaError> {
        let lock = admission.native_origin().original_input_lock().clone();
        manifest.validate(&lock)?;
        let manifest_bytes = serde_json::to_vec(manifest)
            .map_err(|error| invalid(error.to_string()))?
            .len() as u64;
        let output_bytes_max = lock.manifest.output_bytes_max.min(MAX_CORPUS_BYTES);
        let total_bytes = manifest_bytes
            .checked_add(4096)
            .ok_or_else(|| invalid("writer /3 manifest byte count overflow"))?;
        check(
            total_bytes <= output_bytes_max,
            "writer /3 initial output cap",
        )?;
        let manifest_sha256 = manifest.sha256(&lock)?;
        Ok(Self {
            manifest: manifest.clone(),
            index: RoventCorpusIndexLockV3 {
                schema: ROVENT_CORPUS_INDEX_LOCK_V3_DOMAIN.into(),
                manifest_sha256,
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
            uncertain_append: None,
        })
    }

    /// On failure the caller retains the original opaque row and this writer
    /// remains unmodified. No decoded record can issue this capability.
    pub fn append(&mut self, row: &PublishedRoventLockV3Record) -> Result<(), ArenaError> {
        check(
            self.uncertain_append.is_none(),
            "writer /3 append state uncertain",
        )?;
        let admission = row.admission();
        let lock = admission.native_origin().original_input_lock();
        check(lock == &self.lock, "writer /3 input lock differs")?;
        check(
            row.manifest_sha256() == self.index.manifest_sha256,
            "writer /3 manifest differs",
        )?;
        let record = row.record();
        record.validate(&self.manifest, lock)?;
        let value = &record.0;
        check(
            value.native.raw_retirement["root_admission_sha256"] == admission.raw_sha256()
                && value.native.raw_retirement["root_admission"] == *admission.raw_receipt(),
            "writer /3 requires original root admission",
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
            check(
                original.same_original_admission(admission),
                "writer /3 original admission owner changed",
            )?;
        }
        let (snapshot, source_bytes) = if let Some(original) = self.admissions.get(&admission_sha) {
            check(
                self.original_admissions.contains_key(&admission_sha),
                "writer /3 admission custody missing",
            )?;
            (Arc::clone(original), 0)
        } else {
            let snapshot = Arc::new(LockV3PublicationSource::capture(admission)?);
            let source_bytes = (snapshot.lock_bytes().len() as u64)
                .checked_add(1024)
                .ok_or_else(|| invalid("writer /3 source byte count overflow"))?;
            (snapshot, source_bytes)
        };
        let origin_bytes = (serde_json::to_vec(&origin)
            .map_err(|error| invalid(error.to_string()))?
            .len() as u64)
            .checked_add(1)
            .ok_or_else(|| invalid("writer /3 origin byte count overflow"))?;
        let added = (bytes.len() as u64)
            .checked_add(origin_bytes)
            .and_then(|n| n.checked_add(source_bytes))
            .ok_or_else(|| invalid("writer /3 byte count overflow"))?;
        check(
            self.index.origins.len() < MAX_CORPUS_ROWS
                && bytes.len() <= GENERATION_MAX_ROW_BYTES + 1
                && self
                    .buffer
                    .len()
                    .checked_add(bytes.len())
                    .is_some_and(|n| n <= MAX_CHUNK_BYTES)
                && self
                    .total_bytes
                    .checked_add(added)
                    .is_some_and(|n| n <= self.output_bytes_max)
                && !self.record_seals.contains(&value.record_sha256),
            "writer /3 output cap or duplicate; flush explicitly",
        )?;
        self.record_seals.insert(value.record_sha256.clone());
        self.admissions.entry(admission_sha).or_insert(snapshot);
        self.original_admissions
            .entry(admission.raw_sha256().to_owned())
            .or_insert_with(|| admission.clone());
        self.index.origins.push(origin);
        self.index.record_count += 1;
        self.buffer_rows += 1;
        self.total_bytes += added;
        self.buffer.extend(bytes);
        Ok(())
    }

    /// Serialize a prepared target only while the same original ticket/root
    /// gate excludes revoke. The gate is held over the actual memory append,
    /// never over filesystem I/O. The caller retains `pending` in all cases.
    pub(super) fn try_append_policy(
        &mut self,
        pending: &super::producer_v3::PendingPolicyPublicationV3,
    ) -> PolicyAppendResultV3 {
        if self.uncertain_append.is_some() {
            return PolicyAppendResultV3::Uncertain;
        }
        let Some(capability) = pending.target_capability() else {
            return PolicyAppendResultV3::Rejected(invalid("writer /3 target capability absent"));
        };
        // Build the temporary opaque row before acquiring the gate; no caller
        // can retrieve this candidate unless the gate-protected append wins.
        let temporary = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            PublishedRoventLockV3Record {
                record: pending.candidate().clone(),
                manifest_sha256: pending.original_manifest_sha256().to_owned(),
                admission: pending.original_admission().clone(),
            }
        })) {
            Ok(row) => row,
            Err(payload) => return PolicyAppendResultV3::PreparationPanicked(payload),
        };
        let guard = match capability.try_begin_append() {
            Ok(Some(guard)) => guard,
            Ok(None) => return PolicyAppendResultV3::Busy,
            Err(error) => return PolicyAppendResultV3::GateRejected(error),
        };
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.append(&temporary))) {
            Ok(Ok(())) => {
                guard.committed();
                PolicyAppendResultV3::Appended
            }
            Ok(Err(error)) => {
                // All ordinary validation errors in append precede mutation.
                guard.known_not_written();
                PolicyAppendResultV3::Rejected(error)
            }
            Err(payload) => {
                // An unwind may occur after partial mutation. The unresolved
                // guard seals this exact target as unknown on Drop.
                self.uncertain_append = Some(payload);
                drop(guard);
                PolicyAppendResultV3::Uncertain
            }
        }
    }

    pub fn flush(&mut self) -> Result<Option<RoventChunkPacket>, ArenaError> {
        check(
            self.uncertain_append.is_none(),
            "writer /3 append state uncertain",
        )?;
        if self.buffer.is_empty() {
            return Ok(None);
        }
        check(self.index.chunks.len() < 8192, "writer /3 chunk count cap")?;
        let total = self
            .total_bytes
            .checked_add(512)
            .ok_or_else(|| invalid("writer /3 byte count overflow"))?;
        check(total <= self.output_bytes_max, "writer /3 chunk/index cap")?;
        self.index
            .chunks
            .try_reserve(1)
            .map_err(|_| invalid("writer /3 chunk index allocation"))?;
        let declaration = RoventCorpusChunk {
            path: format!("chunks/{:06}.jsonl", self.index.chunks.len()),
            bytes: self.buffer.len() as u64,
            sha256: hex(Digest(Sha256::digest(&self.buffer).into())),
            records: self.buffer_rows,
        };
        // Complete allocation and hashing before moving the sole buffer out.
        let indexed = declaration.clone();
        let bytes = std::mem::take(&mut self.buffer);
        self.total_bytes = total;
        self.buffer_rows = 0;
        self.index.chunks.push(indexed);
        Ok(Some(RoventChunkPacket { declaration, bytes }))
    }

    /// Only assembles the already-flushed corpus. On failure the caller gets
    /// the original writer back; success does not prove physical completion.
    pub fn finish(self) -> Result<FinishedRoventCorpusLockV3, Box<(ArenaError, Self)>> {
        let result = self.validate_finish();
        if let Err(error) = result {
            return Err(Box::new((error, self)));
        }
        Ok(FinishedRoventCorpusLockV3 {
            manifest: self.manifest,
            index: self.index,
            admissions: self.admissions,
        })
    }

    fn validate_finish(&self) -> Result<(), ArenaError> {
        check(
            self.uncertain_append.is_none()
                && self.buffer.is_empty()
                && self.index.record_count > 0
                && !self.admissions.is_empty()
                && self.index.record_count as usize == self.record_seals.len(),
            "writer /3 finish requires flushed original rows",
        )?;
        self.index.validate(&self.manifest, &self.lock)?;
        Ok(())
    }

    /// Validate before moving the original writer; an error or unwind leaves
    /// it in the caller-owned slot. A finished value is assembled evidence,
    /// not a physical worker or durable publication receipt.
    pub(super) fn finish_in_slots(
        owner: &mut Option<Self>,
        finished: &mut Option<FinishedRoventCorpusLockV3>,
    ) -> Result<(), ArenaError> {
        check(finished.is_none(), "writer /3 finished slot occupied")?;
        let current = owner
            .as_ref()
            .ok_or_else(|| invalid("writer /3 owner slot empty"))?;
        current.validate_finish()?;
        let current = owner.take().expect("validated original writer slot");
        *finished = Some(FinishedRoventCorpusLockV3 {
            manifest: current.manifest,
            index: current.index,
            admissions: current.admissions,
        });
        Ok(())
    }
}

/// An assembled corpus only; it cannot recreate live admission or completion.
pub struct FinishedRoventCorpusLockV3 {
    manifest: RoventCorpusManifestLockV3,
    index: RoventCorpusIndexLockV3,
    admissions: BTreeMap<String, Arc<LockV3PublicationSource>>,
}

impl FinishedRoventCorpusLockV3 {
    pub fn manifest(&self) -> &RoventCorpusManifestLockV3 {
        &self.manifest
    }
    pub fn index(&self) -> &RoventCorpusIndexLockV3 {
        &self.index
    }

    /// Uses the original game owner's publication gate. Failure retains
    /// pending material and must not be relabelled as successful publication.
    pub(crate) fn publish_atomic_speculative(
        &self,
        directory: &cap_std::fs::Dir,
        until: std::time::Instant,
        cancel: &std::sync::atomic::AtomicBool,
        game_gate: &std::sync::atomic::AtomicU64,
    ) -> Result<RoventLockV3Publication, ArenaError> {
        publication_lock_v3::publish_lock_v3_payloads_speculative(
            directory,
            until,
            cancel,
            &self.manifest,
            &self.index,
            &self.admissions,
            game_gate,
        )
    }

    /// Internal only: a writer alone has no authority to assert run completion.
    #[allow(dead_code)]
    pub(crate) fn publish_atomic(
        &self,
        directory: &cap_std::fs::Dir,
        until: std::time::Instant,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> Result<RoventLockV3Publication, ArenaError> {
        publish_lock_v3_payloads(
            directory,
            until,
            cancel,
            &self.manifest,
            &self.index,
            &self.admissions,
        )
    }
}
