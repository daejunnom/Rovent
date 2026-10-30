//! The final same-actor ACK callback retains the original completed action
//! before encoding or I/O. These root-local files never create cohort targets.
use super::*;
use rz_uci::pals_adaptive::CompletedUtilityRouteAction;
use rz_uci::rovent_bootstrap::nnuec_source::LoadedRoventNnuecSource;

const MAX_RETAINED_ACTIONS: usize = 64;
const MAX_RETAINED_BYTES: u64 = 128 << 20;

#[derive(Default)]
pub(super) struct UtilityRootLocalStream {
    entries: Vec<OriginalAction>,
    retained_bytes: u64,
}
struct OriginalAction {
    actual: Arc<CompletedUtilityRouteAction>,
    admission: RoventCollectionRootAdmission,
    source: Arc<LoadedRoventNnuecSource>,
    published: Option<PublishedUtilityRootLocalAction>,
    file: Option<RoventPublicationFile>,
}
impl UtilityRootLocalStream {
    pub(super) fn completed(
        &mut self,
        actual: &Arc<CompletedUtilityRouteAction>,
        admission: RoventCollectionRootAdmission,
        source: Arc<LoadedRoventNnuecSource>,
        directory: &Dir,
        timeout: Duration,
        cancel: &AtomicBool,
    ) -> Result<(), ArenaError> {
        if let Some(entry) = self
            .entries
            .iter()
            .find(|entry| entry.actual.id() == actual.id())
        {
            check(
                Arc::ptr_eq(&entry.actual, actual)
                    && entry.actual.digest() == actual.digest()
                    && entry.file.is_some(),
                "utility original action is already retained/incomplete",
            )?;
            return Ok(());
        }
        check(
            self.entries.len() < MAX_RETAINED_ACTIONS,
            "utility root-local retained action capacity",
        )?;
        // Store the source/cap first. Any builder, cap, write or commit error
        // leaves this exact entry in the caller returned by launch failure.
        self.entries.push(OriginalAction {
            actual: Arc::clone(actual),
            admission,
            source,
            published: None,
            file: None,
        });
        let sequence = self.entries.len() as u64;
        let entry = self
            .entries
            .last_mut()
            .expect("stored original utility cap");
        entry.published = Some(publish_utility_root_local_action(
            &entry.actual,
            &entry.admission,
            &entry.source,
            sequence,
        )?);
        let published = entry.published.as_ref().expect("stored utility wire");
        self.retained_bytes = self
            .retained_bytes
            .checked_add(published.bytes().len() as u64)
            .ok_or_else(|| invalid("utility root-local retained byte overflow"))?;
        check(
            self.retained_bytes <= MAX_RETAINED_BYTES,
            "utility root-local retained payload byte capacity",
        )?;
        let until = Instant::now()
            .checked_add(timeout)
            .ok_or_else(|| invalid("utility publication deadline overflow"))?;
        let active = || {
            check(
                Instant::now() < until && !cancel.load(std::sync::atomic::Ordering::Acquire),
                "utility publication canceled/expired; original action retained",
            )
        };
        active()?;
        match directory.create_dir("utility-root-local") {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(io(error)),
        }
        let payloads = directory
            .open_dir_nofollow("utility-root-local")
            .map_err(io)?;
        let name = format!("{sequence:06}.json");
        let pending = format!("{sequence:06}.pending.json");
        let mut options = cap_std::fs::OpenOptions::new();
        options
            .write(true)
            .create_new(true)
            .follow(FollowSymlinks::No);
        let mut file = payloads.open_with(&pending, &options).map_err(io)?;
        for bytes in published.bytes().chunks(64 << 10) {
            active()?;
            file.write_all(bytes).map_err(io)?;
        }
        file.sync_all().map_err(io)?;
        let raw_sha = hex(Digest(Sha256::digest(published.bytes()).into()));
        let pending_pin = RoventPublicationFile {
            path: pending.clone(),
            sha256: raw_sha,
            bytes: published.bytes().len() as u64,
        };
        let readback = read_pin(
            &payloads,
            &pending_pin,
            UTILITY_PUBLICATION_MAX_BYTES as u64,
        )?;
        check(
            readback == published.bytes(),
            "utility original final-action byte readback",
        )?;
        active()?;
        let parent = crate::pals_collect_v3::training_process::directory_file(&payloads)?;
        parent.sync_all().map_err(io)?;
        payloads.hard_link(&pending, &payloads, &name).map_err(io)?;
        payloads.remove_file(&pending).map_err(io)?;
        parent.sync_all().map_err(io)?;
        crate::pals_collect_v3::training_process::directory_file(directory)?
            .sync_all()
            .map_err(io)?;
        entry.file = Some(RoventPublicationFile {
            path: format!("utility-root-local/{name}"),
            ..pending_pin
        });
        Ok(())
    }

    pub(super) fn files(&self, root: &Path) -> Vec<RoventTrainingFilePin> {
        self.entries
            .iter()
            .filter_map(|entry| entry.file.as_ref())
            .map(|pin| RoventTrainingFilePin {
                path: root.join(&pin.path),
                sha256: pin.sha256.clone(),
                bytes: pin.bytes,
            })
            .collect()
    }
}
