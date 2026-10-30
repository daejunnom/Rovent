//! Durable evidence retirement, separate from Store cold-archive receipts.
//! Only the exclusive archive worker writes this stream. An opaque commit
//! attests a completed write/readback barrier for one original retirement;
//! parsing its public metadata cannot mint that commit or a new search right.
use super::{PalsError, ProfiledExternalRetirementIdentity, ProfiledExternalRootRetirement};
use crate::pals::store::{
    ArchiveIoBudget, ArchiveRootCapability, PalsStores, StoreError, StoreReadOwner,
};
use rz_contracts::pals::SearchAuthority;
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::path::Path;
use std::sync::{Arc, atomic::AtomicBool};
use std::time::{Duration, Instant};

mod codec;
mod io;
use codec::{CodecSummary, encode_retirement};
use io::{IoError, IoFailure, IoLimits, IoOwner, IoVerifiedArtifact};

pub const PROFILED_RETIREMENT_SINK_DOMAIN: &str = "rz-rovent-profiled-retirement-sink/1";
pub const PROFILED_RETIREMENT_ARCHIVE_DOMAIN: &str =
    "rz-rovent-profiled-external-retirement-archive/1";
pub const MAX_PROFILED_RETIREMENT_MANAGED_BYTES: u64 = 18 * 1024 * 1024 * 1024;
pub const MAX_PROFILED_RETIREMENT_IO_BYTES: u64 = MAX_PROFILED_RETIREMENT_MANAGED_BYTES * 4;
const MAX_RECORDS: u32 = 262_144;
const MAX_WINDOW_MS: u64 = 3_600_000;

/// A declaration, not filesystem authority or a durable commit. The actual
/// directory capability and original finite admission window are separate.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfiledRetirementSinkRegistration {
    pub domain: String,
    pub max_root_bytes: u64,
    pub max_total_bytes: u64,
    pub max_files: u32,
    pub max_scan_entries: u32,
    pub max_store_records: u32,
    pub max_io_bytes_per_root: u64,
    pub max_persist_ms: u64,
}
impl ProfiledRetirementSinkRegistration {
    pub fn validate(&self) -> Result<(), ProfiledRetirementSinkAdmissionError> {
        if self.domain != PROFILED_RETIREMENT_SINK_DOMAIN
            || self.max_total_bytes == 0
            || self.max_total_bytes > MAX_PROFILED_RETIREMENT_MANAGED_BYTES
            || self.max_root_bytes == 0
            || self.max_root_bytes > self.max_total_bytes / 2
            || !(2..=32_768).contains(&self.max_files)
            || self.max_scan_entries <= self.max_files
            || self.max_scan_entries > 65_536
            || !(1..=MAX_RECORDS).contains(&self.max_store_records)
            || self.max_io_bytes_per_root == 0
            || self.max_io_bytes_per_root > MAX_PROFILED_RETIREMENT_IO_BYTES
            || !(1..=MAX_WINDOW_MS).contains(&self.max_persist_ms)
        {
            return Err(ProfiledRetirementSinkAdmissionError::InvalidRegistration);
        }
        Ok(())
    }
    pub fn canonical_sha256(&self) -> Result<[u8; 32], ProfiledRetirementSinkAdmissionError> {
        self.validate()?;
        let bytes = serde_json::to_vec(&(PROFILED_RETIREMENT_SINK_DOMAIN, self))
            .map_err(|_| ProfiledRetirementSinkAdmissionError::InvalidRegistration)?;
        Ok(Sha256::digest(bytes).into())
    }
}

#[derive(Debug)]
pub enum ProfiledRetirementSinkAdmissionError {
    InvalidRegistration,
    UnsupportedDurablePlatform,
    AdmissionIo { detail: String },
}
impl std::fmt::Display for ProfiledRetirementSinkAdmissionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "profiled retirement sink admission: {self:?}")
    }
}
impl std::error::Error for ProfiledRetirementSinkAdmissionError {}

/// One move-only session owner. Moving it to the existing one-slot I/O worker
/// also moves cumulative quota and failed descriptors; it cannot be cloned to
/// obtain another allowance or reconstructed from the declaration above.
pub struct RegisteredProfiledRetirementSink {
    identity: Arc<()>,
    registration: ProfiledRetirementSinkRegistration,
    registration_sha256: [u8; 32],
    io: IoOwner,
    failure: Option<IoFailure<PalsError>>,
    completion_unknown: bool,
}
impl RegisteredProfiledRetirementSink {
    /// `protected_execution_root` is the caller's verified private input/assets
    /// root. A deployment need not contain a source checkout; this argument
    /// does not claim that the protected directory is a repository.
    /// `admission_scan_deadline` bounds only initial filesystem admission.
    /// Every later persistence uses its retirement's original root cleanup
    /// deadline; this scan window never becomes a session lifetime or renewal.
    pub fn admit(
        registration: ProfiledRetirementSinkRegistration,
        root: ArchiveRootCapability,
        protected_execution_root: &Path,
        admission_scan_deadline: Instant,
        cancel: &AtomicBool,
    ) -> Result<Self, ProfiledRetirementSinkAdmissionError> {
        registration.validate()?;
        if !cfg!(target_os = "linux") {
            return Err(ProfiledRetirementSinkAdmissionError::UnsupportedDurablePlatform);
        }
        let registration_sha256 = registration.canonical_sha256()?;
        let io = IoOwner::new(
            root,
            protected_execution_root,
            IoLimits {
                max_file_bytes: registration.max_root_bytes,
                max_total_bytes: registration.max_total_bytes,
                max_files: registration.max_files,
                max_scan_entries: registration.max_scan_entries,
            },
            admission_scan_deadline,
            cancel,
        )
        .map_err(|error| match error {
            IoError::UnsupportedDurablePlatform => {
                ProfiledRetirementSinkAdmissionError::UnsupportedDurablePlatform
            }
            other => ProfiledRetirementSinkAdmissionError::AdmissionIo {
                detail: other.to_string(),
            },
        })?;
        Ok(Self {
            identity: Arc::new(()),
            registration,
            registration_sha256,
            io,
            failure: None,
            completion_unknown: false,
        })
    }
    pub fn registration(&self) -> &ProfiledRetirementSinkRegistration {
        &self.registration
    }
    pub fn registration_sha256(&self) -> [u8; 32] {
        self.registration_sha256
    }
    pub fn max_io_bytes_per_root(&self) -> u64 {
        self.registration.max_io_bytes_per_root
    }
    pub fn max_persist_duration(&self) -> Duration {
        Duration::from_millis(self.registration.max_persist_ms)
    }
    pub fn same_sink(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.identity, &other.identity)
    }
    pub fn admission_closed(&self) -> bool {
        self.failure.is_some() || self.completion_unknown || self.io.observation().poisoned
    }
    /// Data I/O only. Metadata scan syscalls and allocator/RSS costs are not
    /// converted into bytes. An ambiguous data operation remains unknown.
    pub fn data_io_bytes(&self) -> Option<u64> {
        self.io.data_io_bytes()
    }
    pub fn observation(&self) -> serde_json::Value {
        let io = self.io.observation();
        serde_json::json!({"domain":"rz-rovent-profiled-retirement-sink-observation/1",
            "registration_sha256":hex(self.registration_sha256),"session_written_bytes":io.session_written_bytes,
            "session_reserved_bytes":io.session_reserved_bytes,"managed_file_extent_bytes":io.managed_disk_bytes,
            "managed_files":io.managed_files,"verified_artifacts":io.verified_artifacts,
            "admission_closed":self.admission_closed(),"failed_file_retained":io.failed_file_retained,
            "lock_retained":io.lock_retained,"completion_unknown":self.completion_unknown,
            "data_io_bytes":self.data_io_bytes(),"host_owned_bytes":null,
            "host_measurement":"unknown_not_serialized_file_extent",
            "failure":self.failure.as_ref().map(|error|format!("{error:?}")),
            "past_native_authority_reissued":false})
    }
    pub(crate) fn validate_owner(
        &self,
        stores: &PalsStores,
        retirement: &ProfiledExternalRootRetirement,
        authority: SearchAuthority,
        deadline: Instant,
        max_io_bytes: u64,
    ) -> Result<(), PalsError> {
        validate_persist_window(
            retirement
                .original_cleanup_deadline()
                .ok_or(StoreError::InvalidConditions(
                    "missing original root cleanup deadline",
                ))?,
            self.max_persist_duration(),
            Instant::now(),
            deadline,
        )?;
        if self.admission_closed() {
            return Err(StoreError::ArchiveIntegrity("retirement sink admission closed").into());
        }
        if retirement.authority() != authority {
            return Err(StoreError::StaleConsumer.into());
        }
        retirement.validate_owner(stores)?;
        retirement.protected_pins()?;
        if max_io_bytes == 0 || max_io_bytes > self.max_io_bytes_per_root() {
            return Err(StoreError::ArchiveByteBudget.into());
        }
        Ok(())
    }
    /// Called only while the original exclusive Engine, Store and retirement
    /// payload reside outside any unwind closure. No semantic state changes.
    pub(crate) fn persist(
        &mut self,
        stores: &PalsStores,
        retirement: &ProfiledExternalRootRetirement,
        budget: &mut ArchiveIoBudget,
        cancel: &AtomicBool,
    ) -> Result<ProfiledRetirementCommit, PalsError> {
        self.validate_owner(
            stores,
            retirement,
            retirement.authority(),
            budget.deadline,
            budget.max_bytes,
        )?;
        let store_owner = stores.read_owner();
        let generation = stores.generation();
        let retirement_identity = retirement.identity();
        let before = self.data_io_bytes();
        let registration = &self.registration;
        let root_cleanup_deadline =
            retirement
                .original_cleanup_deadline()
                .ok_or(StoreError::InvalidConditions(
                    "missing original root cleanup deadline",
                ))?;
        let deadline = budget.deadline;
        let result = self
            .io
            .write_stream(deadline, cancel, budget.max_bytes, |writer| {
                encode_retirement(
                    writer,
                    stores,
                    retirement,
                    registration,
                    root_cleanup_deadline,
                    deadline,
                    cancel,
                )
            });
        match (before, self.data_io_bytes()) {
            (Some(before), Some(after)) => {
                budget.max_bytes = after
                    .checked_sub(before)
                    .and_then(|delta| budget.max_bytes.checked_sub(delta))
                    .unwrap_or(0);
            }
            _ => budget.max_bytes = 0,
        }
        match result {
            Ok(artifact) => Ok(ProfiledRetirementCommit {
                sink_identity: Arc::clone(&self.identity),
                retirement_identity,
                store_owner,
                generation,
                authority: retirement.authority(),
                registration_sha256: self.registration_sha256,
                artifact,
            }),
            Err(error) => {
                let outward = failure_error(&error);
                self.failure = Some(error);
                Err(outward)
            }
        }
    }
    pub(crate) fn validate_commit(
        &self,
        stores: &PalsStores,
        retirement: &ProfiledExternalRootRetirement,
        commit: &ProfiledRetirementCommit,
    ) -> Result<(), PalsError> {
        if !Arc::ptr_eq(&self.identity, &commit.sink_identity)
            || self.registration_sha256 != commit.registration_sha256
        {
            return Err(StoreError::StaleConsumer.into());
        }
        commit.matches_retirement(stores, retirement)
    }
    pub(crate) fn quarantine_completion_unknown(&mut self) {
        self.completion_unknown = true;
    }
}

/// The original stream FD and opaque identities are private and non-Clone.
/// This is an in-process acknowledgement of actual durable I/O, never a JSON
/// capability, new native execution, successful search or reusable root token.
pub struct ProfiledRetirementCommit {
    sink_identity: Arc<()>,
    retirement_identity: ProfiledExternalRetirementIdentity,
    store_owner: StoreReadOwner,
    generation: u64,
    authority: SearchAuthority,
    registration_sha256: [u8; 32],
    artifact: IoVerifiedArtifact<CodecSummary>,
}
impl ProfiledRetirementCommit {
    /// Pure owner matching on the manager. All filesystem readback/sync work
    /// was performed by the archive worker before this private proof existed.
    pub fn matches_retirement(
        &self,
        stores: &PalsStores,
        retirement: &ProfiledExternalRootRetirement,
    ) -> Result<(), PalsError> {
        if !self.store_owner.matches(stores)
            || self.generation != stores.generation()
            || self.authority != retirement.authority()
            || !retirement.matches_identity(&self.retirement_identity)
            || self.artifact.summary().packet_count != retirement.retained_questions() as u64
        {
            return Err(StoreError::StaleConsumer.into());
        }
        retirement.validate_owner(stores)
    }
    pub fn relative_path(&self) -> &str {
        self.artifact.name()
    }
    pub fn sha256(&self) -> [u8; 32] {
        self.artifact.sha256()
    }
    pub fn bytes(&self) -> u64 {
        self.artifact.bytes()
    }
    pub fn observation(&self) -> serde_json::Value {
        let summary = self.artifact.summary();
        serde_json::json!({"domain":"rz-rovent-profiled-retirement-commit/1",
            "archive_domain":PROFILED_RETIREMENT_ARCHIVE_DOMAIN,"relative_path":self.relative_path(),
            "sha256":hex(self.sha256()),"bytes":self.bytes(),"registration_sha256":hex(self.registration_sha256),
            "packet_count":summary.packet_count,"store_records":summary.store_records,
            "stream_records":summary.stream_records,"source_error_count":summary.source_error_count,
            "data_io_bytes":self.artifact.io_bytes(),
            "durable_file_and_directory_sync":true,"readback_verified":true,
            "past_native_authority_reissued":false,"historical_evidence_only":true})
    }
}
fn failure_error(error: &IoFailure<PalsError>) -> PalsError {
    match error {
        IoFailure::Encode { error, io: None } => error.clone(),
        IoFailure::Encode {
            io: Some(error), ..
        }
        | IoFailure::Io(error) => match error {
            IoError::UnsupportedDurablePlatform => {
                StoreError::InvalidConditions("unsupported durable retirement platform").into()
            }
            IoError::Boundary(detail) => StoreError::ArchiveIntegrity(detail).into(),
            IoError::Cancelled => StoreError::ArchiveCanceled.into(),
            IoError::Deadline => StoreError::ArchiveDeadline.into(),
            IoError::ByteBudget => StoreError::ArchiveByteBudget.into(),
            IoError::Store(error) => error.clone().into(),
            IoError::Io { stage, error } => StoreError::ArchiveIo {
                stage,
                kind: error.kind(),
            }
            .into(),
        },
    }
}
fn hex(bytes: [u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn validate_persist_window(
    original_until: Instant,
    max_persist: Duration,
    now: Instant,
    deadline: Instant,
) -> Result<(), PalsError> {
    if now >= deadline || deadline > original_until || deadline.duration_since(now) > max_persist {
        return Err(StoreError::ArchiveDeadline.into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retirement_window_cannot_extend_original_owner_deadline() {
        let now = Instant::now();
        let original_until = now + Duration::from_secs(2);
        assert!(
            validate_persist_window(
                original_until,
                Duration::from_secs(10),
                now,
                original_until + Duration::from_nanos(1),
            )
            .is_err()
        );
        assert!(
            validate_persist_window(original_until, Duration::from_secs(10), now, original_until,)
                .is_ok()
        );
    }

    #[test]
    fn retirement_window_keeps_per_operation_limit_and_expiry() {
        let now = Instant::now();
        let original_until = now + Duration::from_secs(10);
        assert!(
            validate_persist_window(
                original_until,
                Duration::from_secs(1),
                now,
                now + Duration::from_secs(2),
            )
            .is_err()
        );
        assert!(
            validate_persist_window(
                original_until,
                Duration::from_secs(1),
                original_until,
                original_until,
            )
            .is_err()
        );
    }
}
