//! Bounded, failure-retaining assembly of original lock /2 rows.
//!
//! Assembly is deliberately separate from whole-collection publication: a
//! flushed row and an index do not establish a native fence or teacher drain.
//! Schedule assembly only after the original observer has settled and while no
//! next root can invoke it: the producer uses the same lock as its callbacks.
use super::*;
use cap_fs_ext::{DirExt, OpenOptionsFollowExt};
use cap_std::fs::Dir;
use sha2::Digest as _;
use std::{io::Write, path::Component};

/// The caller owns its directory, writer, and at most one row or chunk in
/// each transition. A failed transition leaves its original objects here.
pub struct RoventCorpusAssemblyLockV2 {
    owner: RoventCollectionOwnerLockV2,
    directory: Dir,
    writer: Option<RoventCorpusWriterLockV2>,
    pending: Option<RoventRetiredLockV2Record>,
    sealed: Option<PublishedRoventLockV2Record>,
    chunk: Option<RoventChunkPacket>,
    chunk_write_attempted: bool,
    finished: Option<FinishedRoventCorpusLockV2>,
    records: u64,
    maximum_records: u64,
}

impl RoventCorpusAssemblyLockV2 {
    /// The writer is bound to the original admitted lock, not to reconstructed
    /// receipt bytes. The directory must already be owned by the caller.
    pub fn new(
        owner: &RoventCollectionOwnerLockV2,
        directory: &Dir,
        manifest: &rz_experiments::rovent_corpus_lock_v2::record::RoventCorpusManifestLockV2,
        admission: &rz_uci::pals_adaptive::RoventCollectionRootAdmissionV2,
        maximum_records: u64,
    ) -> Result<Self, ArenaError> {
        check(maximum_records > 0, "lock-v2 assembly record cap is zero")?;
        Ok(Self {
            owner: owner.clone(),
            directory: directory
                .try_clone()
                .map_err(|error| invalid(error.to_string()))?,
            writer: Some(RoventCorpusWriterLockV2::new(manifest, admission)?),
            pending: None,
            sealed: None,
            chunk: None,
            chunk_write_attempted: false,
            finished: None,
            records: 0,
            maximum_records,
        })
    }

    /// A rejected offer returns the identical in-process row to the caller.
    pub fn offer(
        &mut self,
        row: RoventRetiredLockV2Record,
    ) -> Result<(), Box<(ArenaError, RoventRetiredLockV2Record)>> {
        if self.writer.is_none()
            || self.finished.is_some()
            || self.pending.is_some()
            || self.sealed.is_some()
            || self.chunk.is_some()
            || self.records >= self.maximum_records
        {
            return Err(Box::new((
                invalid("lock-v2 assembly busy, closed or at record cap"),
                row,
            )));
        }
        self.pending = Some(row);
        Ok(())
    }

    /// Give a target attachment operation the same original producer and row.
    /// Attachment is not evidence of teacher execution or physical completion.
    /// The callback must be bounded and must not perform teacher I/O or wait
    /// for another worker while holding the original producer lock.
    pub fn with_pending<T>(
        &mut self,
        action: impl FnOnce(
            &mut RoventCorpusProducerLockV2,
            &mut RoventRetiredLockV2Record,
        ) -> Result<T, ArenaError>,
    ) -> Result<T, ArenaError> {
        let row = self
            .pending
            .as_mut()
            .ok_or_else(|| invalid("lock-v2 pending row absent"))?;
        self.owner.with_retired_producer_mut(row, action)
    }

    /// Validate and append one original row to the bounded writer. On failure
    /// the original pending or sealed row remains available. If the writer's
    /// chunk is full, the caller may explicitly flush and retry the same row.
    pub fn advance(&mut self) -> Result<(), ArenaError> {
        check(
            self.finished.is_none() && self.chunk.is_none(),
            "lock-v2 assembly closed or chunk pending",
        )?;
        if self.pending.is_some() {
            self.with_pending(|producer, row| producer.seal_retired(row))?;
            let row = self.pending.take().expect("pending row was just sealed");
            self.sealed = Some(PublishedRoventLockV2Record {
                record: row.record,
                manifest_sha256: row.manifest_sha256,
                admission: row.admission,
            });
        }
        if let Some(row) = self.sealed.as_ref() {
            let next_records = self
                .records
                .checked_add(1)
                .ok_or_else(|| invalid("lock-v2 record count overflow"))?;
            self.writer
                .as_mut()
                .ok_or_else(|| invalid("lock-v2 writer missing"))?
                .append(row)?;
            self.sealed = None;
            self.records = next_records;
        }
        Ok(())
    }

    /// Persist the current bounded chunk. Failures retain the original packet;
    /// a partial filesystem write is never silently retried.
    pub fn flush(&mut self) -> Result<(), ArenaError> {
        check(self.finished.is_none(), "lock-v2 assembly already finished")?;
        if self.chunk.is_none() {
            self.chunk = self
                .writer
                .as_mut()
                .ok_or_else(|| invalid("lock-v2 writer missing"))?
                .flush()?;
            self.chunk_write_attempted = false;
        }
        if let Some(packet) = self.chunk.as_ref() {
            check(
                !self.chunk_write_attempted,
                "lock-v2 failed chunk write requires explicit recovery",
            )?;
            let path = std::path::Path::new(&packet.declaration.path);
            let mut components = path.components();
            let leaf = match (components.next(), components.next(), components.next()) {
                (Some(Component::Normal(first)), Some(Component::Normal(leaf)), None)
                    if first == "chunks" =>
                {
                    leaf
                }
                _ => return Err(invalid("lock-v2 chunk path is not a chunks leaf")),
            };
            check(
                !path.is_absolute()
                    && packet.bytes.len() as u64 == packet.declaration.bytes
                    && hex(Digest(Sha256::digest(&packet.bytes).into()))
                        == packet.declaration.sha256,
                "lock-v2 chunk packet changed",
            )?;
            let chunks = self
                .directory
                .open_dir_nofollow("chunks")
                .map_err(|error| invalid(error.to_string()))?;
            let mut options = cap_std::fs::OpenOptions::new();
            options
                .write(true)
                .create_new(true)
                .follow(cap_fs_ext::FollowSymlinks::No);
            self.chunk_write_attempted = true;
            let mut file = chunks
                .open_with(leaf, &options)
                .map_err(|error| invalid(error.to_string()))?;
            file.write_all(&packet.bytes)
                .and_then(|_| file.sync_all())
                .map_err(|error| invalid(error.to_string()))?;
            self.chunk = None;
            self.chunk_write_attempted = false;
        }
        Ok(())
    }

    /// The assembled corpus is partial evidence only. An independent owner
    /// must still establish native physical completion and teacher terminals.
    pub fn finish(&mut self) -> Result<&FinishedRoventCorpusLockV2, ArenaError> {
        check(
            self.pending.is_none() && self.sealed.is_none() && self.chunk.is_none(),
            "lock-v2 assembly retains unfinished row or chunk",
        )?;
        if self.finished.is_none() {
            let writer = self
                .writer
                .take()
                .ok_or_else(|| invalid("lock-v2 writer missing"))?;
            match writer.finish() {
                Ok(finished) => self.finished = Some(finished),
                Err(failure) => {
                    let (error, writer) = *failure;
                    self.writer = Some(writer);
                    return Err(error);
                }
            }
        }
        self.finished
            .as_ref()
            .ok_or_else(|| invalid("lock-v2 finished assembly missing"))
    }

    pub fn records(&self) -> u64 {
        self.records
    }
    pub fn pending_row(&self) -> Option<&RoventRetiredLockV2Record> {
        self.pending.as_ref()
    }
    pub fn sealed_row(&self) -> Option<&PublishedRoventLockV2Record> {
        self.sealed.as_ref()
    }
    pub fn pending_chunk(&self) -> Option<&RoventChunkPacket> {
        self.chunk.as_ref()
    }
    pub fn chunk_write_attempted(&self) -> bool {
        self.chunk_write_attempted
    }
}
