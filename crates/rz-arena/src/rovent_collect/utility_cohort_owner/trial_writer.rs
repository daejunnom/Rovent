//! Actual trial checkpoint publication. The retained original retirement and
//! same UCI owner are required; detached DTOs cannot enter this writer.
use super::*;
use cap_fs_ext::{DirExt, FollowSymlinks, OpenOptionsFollowExt};
use cap_std::fs::Dir;
use rz_uci::pals_adaptive::{
    RoventCollectionRootAdmission, UtilityCohortDurableAck, UtilityComparisonCohortOwner,
    UtilityTrialPhysicalRetirement,
};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Instant;

/// Failure leaves the original final cap, exact bytes, failed FD and any
/// already issued ACK in this owner. Only an accepted actual durable ACK lets
/// UCI admit the next independent trial.
pub struct RoventUtilityCohortTrialWriter {
    directory: Dir,
    cohort: Option<Digest>,
    originals: Vec<OriginalTrial>,
    payload_bytes: u64,
    failure: Option<String>,
}
struct OriginalTrial {
    original: Arc<UtilityTrialPhysicalRetirement>,
    materialization: Option<RoventPublicationFile>,
    checkpoint: Option<RoventPublicationFile>,
    fd: Option<File>,
    ack: Option<Arc<UtilityCohortDurableAck>>,
    observation: Option<RoventUtilityCohortTrialAcknowledgement>,
    accepted: bool,
}
impl RoventUtilityCohortTrialWriter {
    pub fn new(directory: Dir) -> Self {
        Self {
            directory,
            cohort: None,
            originals: Vec::new(),
            payload_bytes: 0,
            failure: None,
        }
    }

    pub fn publish_trial_retaining(
        &mut self,
        original: &Arc<UtilityTrialPhysicalRetirement>,
        owner: &mut UtilityComparisonCohortOwner,
        admission: &RoventCollectionRootAdmission,
        until: Instant,
        cancel: &AtomicBool,
    ) -> Result<(), ArenaError> {
        check(
            self.failure.is_none(),
            "utility cohort original publication failure retained",
        )?;
        if let Some(previous) = self
            .originals
            .iter()
            .find(|entry| entry.original.trial_sequence() == original.trial_sequence())
        {
            return check(
                Arc::ptr_eq(&previous.original, original) && previous.accepted,
                "utility cohort duplicate/unfinished original trial",
            );
        }
        check(
            self.originals.len() < 3
                && original.cohort_id() == owner.cohort_id()
                && original.trial_sequence() == self.originals.len() as u64 + 1
                && self
                    .cohort
                    .is_none_or(|cohort| cohort == original.cohort_id()),
            "utility cohort original owner/order/capacity",
        )?;
        self.cohort = Some(original.cohort_id());
        self.originals.push(OriginalTrial {
            original: Arc::clone(original),
            materialization: None,
            checkpoint: None,
            fd: None,
            ack: None,
            observation: None,
            accepted: false,
        });
        let result = self.publish_last(owner, admission, until, cancel);
        if let Err(error) = &result {
            self.failure = Some(error.to_string());
        }
        result
    }

    fn publish_last(
        &mut self,
        owner: &mut UtilityComparisonCohortOwner,
        admission: &RoventCollectionRootAdmission,
        until: Instant,
        cancel: &AtomicBool,
    ) -> Result<(), ArenaError> {
        let entry = self.originals.last_mut().expect("retained original trial");
        let original = &entry.original;
        check(
            original.matches_root_admission(admission.root_admission()),
            "utility cohort original root/clock differs",
        )?;
        let clock = admission.root_admission().runtime_clock();
        let started_tick_ns = clock
            .try_now()
            .map_err(|error| invalid(format!("{error:?}")))?
            .0;
        let ordinal = original.trial_sequence() - 1;
        let bytes = original.original_bytes();
        let materialized = original.materialized();
        let added = (bytes.len() as u64)
            .checked_add(materialized.map_or(0, |m| m.original_bytes().len() as u64))
            .ok_or_else(|| invalid("utility cohort original payload size overflow"))?;
        self.payload_bytes = self
            .payload_bytes
            .checked_add(added)
            .ok_or_else(|| invalid("utility cohort retained payload size overflow"))?;
        check(
            !bytes.is_empty()
                && bytes.len() as u64 <= ROVENT_UTILITY_COHORT_MATERIALIZATION_MAX
                && self.payload_bytes <= ROVENT_UTILITY_COHORT_PAYLOAD_MAX,
            "utility cohort bounded original checkpoint/materialization",
        )?;
        if let Some(materialized) = materialized {
            check(
                materialized.trial_sequence() == original.trial_sequence()
                    && materialized.cohort_id() == original.cohort_id()
                    && materialized.original_bytes().len() as u64
                        <= ROVENT_UTILITY_COHORT_MATERIALIZATION_MAX,
                "utility cohort original materialization binding",
            )?;
            entry.materialization = Some(put_original(
                &self.directory,
                "materializations",
                ordinal,
                materialized.original_bytes(),
                &mut entry.fd,
                until,
                cancel,
            )?);
            entry.fd = None;
        }
        entry.checkpoint = Some(put_original(
            &self.directory,
            "trials",
            ordinal,
            bytes,
            &mut entry.fd,
            until,
            cancel,
        )?);
        let file = entry.fd.take().expect("retained original checkpoint FD");
        let actual_ack =
            match UtilityCohortDurableAck::acknowledge_trial_publication(file, original) {
                Ok(ack) => Arc::new(ack),
                Err(rejected) => {
                    entry.fd = Some(rejected.file);
                    return Err(invalid(format!("{:?}", rejected.error)));
                }
            };
        // Retain the real FD-bearing ACK before the owner checks it. The
        // original cap and ACK remain available even if that final join fails.
        entry.ack = Some(Arc::clone(&actual_ack));
        let finished_tick_ns = clock
            .try_now()
            .map_err(|error| invalid(format!("{error:?}")))?
            .0;
        let materialized_bytes = entry.materialization.as_ref().map_or(0, |pin| pin.bytes);
        let checkpoint = entry.checkpoint.as_ref().expect("original checkpoint pin");
        entry.observation = Some(RoventUtilityCohortTrialAcknowledgement {
            trial_sequence: original.trial_sequence(),
            native_trial_sha256: hex(actual_ack.trial_sha256()),
            checkpoint: checkpoint.clone(),
            collection_io: RoventUtilityCohortCollectionIo {
                clock_epoch: admission.root_admission().hard_deadline().clock.0.0,
                started_tick_ns,
                finished_tick_ns,
                elapsed_ns: finished_tick_ns
                    .checked_sub(started_tick_ns)
                    .ok_or_else(|| invalid("utility cohort original I/O clock reversed"))?,
                written_bytes: materialized_bytes + checkpoint.bytes,
                readback_bytes: materialized_bytes + 2 * checkpoint.bytes,
                synced_files: 2 + u64::from(materialized_bytes > 0),
                synced_directories: 3 * (1 + u64::from(materialized_bytes > 0)),
            },
        });
        owner
            .accept_durable_ack(original, actual_ack)
            .map_err(|rejected| invalid(format!("{:?}", rejected.error)))?;
        entry.accepted = true;
        Ok(())
    }

    pub fn original_trials(&self) -> impl Iterator<Item = &Arc<UtilityTrialPhysicalRetirement>> {
        self.originals.iter().map(|entry| &entry.original)
    }
    pub fn materialization_files(&self) -> impl Iterator<Item = &RoventPublicationFile> {
        self.originals
            .iter()
            .filter_map(|entry| entry.materialization.as_ref())
    }
    pub fn trial_files(&self) -> impl Iterator<Item = &RoventPublicationFile> {
        self.originals
            .iter()
            .filter_map(|entry| entry.checkpoint.as_ref())
    }
    pub fn original_acks(&self) -> impl Iterator<Item = &Arc<UtilityCohortDurableAck>> {
        self.originals.iter().filter_map(|entry| entry.ack.as_ref())
    }
    pub fn acknowledgement_observations(
        &self,
    ) -> impl Iterator<Item = &RoventUtilityCohortTrialAcknowledgement> {
        self.originals
            .iter()
            .filter_map(|entry| entry.observation.as_ref())
    }
    pub fn failure(&self) -> Option<&str> {
        self.failure.as_deref()
    }
}

fn io(error: std::io::Error) -> ArenaError {
    ArenaError::Io(error.to_string())
}
fn put_original(
    root: &Dir,
    subdirectory: &str,
    ordinal: u64,
    bytes: &[u8],
    retained: &mut Option<File>,
    until: Instant,
    cancel: &AtomicBool,
) -> Result<RoventPublicationFile, ArenaError> {
    let active = || {
        check(
            Instant::now() < until && !cancel.load(Ordering::Acquire),
            "utility cohort publication canceled/expired; source retained",
        )
    };
    active()?;
    check(
        retained.is_none(),
        "utility cohort failed payload FD retained",
    )?;
    match root.create_dir(subdirectory) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(io(error)),
    }
    let directory = root.open_dir_nofollow(subdirectory).map_err(io)?;
    let name = format!("{ordinal:06}.json");
    let pending = format!("{ordinal:06}.pending.json");
    let mut options = cap_std::fs::OpenOptions::new();
    options
        .read(true)
        .write(true)
        .create_new(true)
        .follow(FollowSymlinks::No);
    *retained = Some(
        directory
            .open_with(&pending, &options)
            .map_err(io)?
            .into_std(),
    );
    let file = retained.as_mut().expect("stored original payload FD");
    for part in bytes.chunks(64 << 10) {
        active()?;
        file.write_all(part).map_err(io)?;
    }
    file.sync_all().map_err(io)?;
    file.seek(SeekFrom::Start(0)).map_err(io)?;
    let mut buffer = [0u8; 8192];
    for part in bytes.chunks(buffer.len()) {
        active()?;
        file.read_exact(&mut buffer[..part.len()]).map_err(io)?;
        check(
            buffer[..part.len()] == *part,
            "utility cohort original payload readback",
        )?;
    }
    check(
        file.read(&mut buffer[..1]).map_err(io)? == 0,
        "utility cohort payload grew",
    )?;
    active()?;
    let directory_fd = crate::pals_collect_v3::training_process::directory_file(&directory)?;
    directory_fd.sync_all().map_err(io)?;
    directory
        .hard_link(&pending, &directory, &name)
        .map_err(io)?;
    directory.remove_file(&pending).map_err(io)?;
    directory_fd.sync_all().map_err(io)?;
    crate::pals_collect_v3::training_process::directory_file(root)?
        .sync_all()
        .map_err(io)?;
    Ok(RoventPublicationFile {
        path: format!("{subdirectory}/{name}"),
        sha256: hex(Digest(Sha256::digest(bytes).into())),
        bytes: bytes.len() as u64,
    })
}
