//! /2 전용 원 FD와 누적 read budget을 보존하는 역사 소비자.
//! 이미 열린 FD만 검증하며 원 경로를 재개방하거나 budget을 갱신하지 않는다.
use super::*;
use crate::rovent_training_process::{RoventTrainingFilePin, recovery::RoventRecoveryReadBudget};
use cap_fs_ext::{DirExt, FollowSymlinks, OpenOptionsFollowExt};
use cap_std::fs::{Dir, OpenOptionsExt};
use std::{
    fs::File,
    os::unix::fs::MetadataExt,
    path::{Path, PathBuf},
    sync::Arc,
};

struct OriginalFile {
    pin: RoventPublicationFile,
    file: File,
    identity: Option<(u64, u64)>,
}
/// 성공과 실패 모두 원 directory/FD/budget을 보존한다. JSON은 실행 권한이 아니다.
pub struct RoventHistoricalUtilityInterventionFiles {
    root: PathBuf,
    directory: Option<Dir>,
    budget: Arc<RoventRecoveryReadBudget>,
    held: Vec<OriginalFile>,
    payloads: BTreeMap<String, Vec<u8>>,
    publication_raw: Option<Vec<u8>>,
    ack_raw: Option<Vec<u8>>,
    publication: Option<RoventUtilityInterventionPublication>,
    ack: Option<RoventUtilityInterventionDurableAck>,
    evidence: Option<RoventHistoricalUtilityInterventionEvidence>,
}
pub struct RoventUtilityInterventionHistoricalReadRejected {
    pub error: ArenaError,
    pub original: RoventHistoricalUtilityInterventionFiles,
}
impl std::fmt::Debug for RoventUtilityInterventionHistoricalReadRejected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RoventUtilityInterventionHistoricalReadRejected")
            .field("error", &self.error)
            .field("original_files_retained", &self.original.held.len())
            .finish_non_exhaustive()
    }
}
impl std::fmt::Display for RoventUtilityInterventionHistoricalReadRejected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.error.fmt(f)
    }
}
impl std::error::Error for RoventUtilityInterventionHistoricalReadRejected {}
impl RoventHistoricalUtilityInterventionFiles {
    /// `budget`의 원 deadline과 누적 read credit을 그대로 사용한다.
    /// 실패는 열린 원 FD, 완료/부분 자료, budget을 caller에게 돌려준다.
    pub fn open_retaining(
        root: PathBuf,
        publication_pin: RoventPublicationFile,
        ack_pin: RoventPublicationFile,
        budget: Arc<RoventRecoveryReadBudget>,
    ) -> Result<Self, Box<RoventUtilityInterventionHistoricalReadRejected>> {
        let mut original = Self {
            root,
            directory: None,
            budget,
            held: Vec::new(),
            payloads: BTreeMap::new(),
            publication_raw: None,
            ack_raw: None,
            publication: None,
            ack: None,
            evidence: None,
        };
        let checked = (|| {
            check(
                original.root.is_absolute()
                    && original.root.components().all(|component| {
                        !matches!(
                            component,
                            std::path::Component::CurDir | std::path::Component::ParentDir
                        )
                    })
                    && original.root.file_name()
                        == Some(std::ffi::OsStr::new("utility-intervention-cohort"))
                    && publication_pin.path == ROVENT_UTILITY_INTERVENTION_PUBLICATION_FILE
                    && ack_pin.path == ROVENT_UTILITY_INTERVENTION_ACK_FILE,
                "explicit /2 historical root/publication/ACK required",
            )?;
            let parent = original
                .root
                .parent()
                .ok_or_else(|| invalid("original /2 root parent absent"))?;
            let parent = Dir::open_ambient_dir(parent, cap_std::ambient_authority())
                .map_err(|e| ArenaError::Io(e.to_string()))?;
            original.directory = Some(
                parent
                    .open_dir_nofollow("utility-intervention-cohort")
                    .map_err(|e| ArenaError::Io(e.to_string()))?,
            );
            original.publication_raw = Some(original.read_original(
                &publication_pin,
                ROVENT_UTILITY_INTERVENTION_METADATA_MAX as u64,
            )?);
            let publication = RoventUtilityInterventionPublication::from_json_historical(
                std::str::from_utf8(
                    original
                        .publication_raw
                        .as_ref()
                        .expect("held original publication"),
                )
                .map_err(|_| invalid("/2 historical publication UTF8"))?,
            )?;
            original.publication = Some(publication);
            original.ack_raw = Some(
                original
                    .read_original(&ack_pin, ROVENT_UTILITY_INTERVENTION_METADATA_MAX as u64)?,
            );
            let publication = original.publication.as_ref().expect("held publication");
            original.ack = Some(RoventUtilityInterventionDurableAck::from_json_historical(
                std::str::from_utf8(original.ack_raw.as_ref().expect("held original ACK"))
                    .map_err(|_| invalid("/2 historical ACK UTF8"))?,
                publication,
                &publication_pin,
            )?);
            let payload_pins = publication.payloads().cloned().collect::<Vec<_>>();
            for pin in payload_pins {
                let raw = original.read_original(&pin, ROVENT_UTILITY_INTERVENTION_PAYLOAD_MAX)?;
                check(
                    original.payloads.insert(pin.path.clone(), raw).is_none(),
                    "duplicate /2 original payload",
                )?;
            }
            let publication = original.publication.as_ref().expect("held publication");
            let ack = original.ack.as_ref().expect("held ACK");
            ack.validate_original_trial_io(publication, &original.payloads)?;
            original.evidence = Some(verify_historical_utility_intervention(
                publication,
                &original.payloads,
            )?);
            Ok(())
        })();
        match checked {
            Ok(()) => Ok(original),
            Err(error) => Err(Box::new(RoventUtilityInterventionHistoricalReadRejected {
                error,
                original,
            })),
        }
    }
    fn read_original(
        &mut self,
        pin: &RoventPublicationFile,
        maximum: u64,
    ) -> Result<Vec<u8>, ArenaError> {
        pin.validate()?;
        check(
            pin.bytes <= maximum
                && self.held.len() < 32
                && !self.held.iter().any(|held| held.pin.path == pin.path),
            "/2 original FD path/size/count",
        )?;
        let root = self
            .directory
            .as_ref()
            .ok_or_else(|| invalid("original /2 directory absent"))?;
        let (directory, name) = match pin.path.rsplit_once('/') {
            Some((parent, name)) => {
                check(
                    matches!(parent, "trials" | "materializations" | "pairs"),
                    "original /2 payload directory",
                )?;
                (
                    root.open_dir_nofollow(parent)
                        .map_err(|e| ArenaError::Io(e.to_string()))?,
                    name,
                )
            }
            None => (
                root.try_clone()
                    .map_err(|e| ArenaError::Io(e.to_string()))?,
                pin.path.as_str(),
            ),
        };
        let mut options = cap_std::fs::OpenOptions::new();
        options
            .read(true)
            .follow(FollowSymlinks::No)
            .custom_flags(libc::O_CLOEXEC | libc::O_NONBLOCK);
        let file = directory
            .open_with(name, &options)
            .map_err(|e| ArenaError::Io(e.to_string()))?
            .into_std();
        self.held.push(OriginalFile {
            pin: pin.clone(),
            file,
            identity: None,
        });
        let held = self
            .held
            .last_mut()
            .expect("original FD retained before validation");
        let metadata = held
            .file
            .metadata()
            .map_err(|e| ArenaError::Io(e.to_string()))?;
        held.identity = Some((metadata.dev(), metadata.ino()));
        check(metadata.is_file(), "original /2 payload must be regular")?;
        let absolute_pin = RoventTrainingFilePin {
            path: self.root.join(Path::new(&pin.path)),
            sha256: pin.sha256.clone(),
            bytes: pin.bytes,
        };
        self.budget.verify_held(&held.file, &absolute_pin, true)
    }
    /// Recheck only the already retained descriptors with the caller's current
    /// finite read budget. The original owner/budget stays alive and no pathname
    /// is reopened; failed/partial read credit is never refunded.
    pub fn revalidate_original(&self, budget: &RoventRecoveryReadBudget) -> Result<(), ArenaError> {
        check(
            self.directory.is_some()
                && self.evidence.is_some()
                && self.publication.is_some()
                && self.ack.is_some(),
            "/2 successful original FD owner required",
        )?;
        for held in &self.held {
            let original = held
                .identity
                .ok_or_else(|| invalid("/2 original FD identity unavailable"))?;
            let pin = RoventTrainingFilePin {
                path: self.root.join(Path::new(&held.pin.path)),
                sha256: held.pin.sha256.clone(),
                bytes: held.pin.bytes,
            };
            let raw = budget.verify_held(&held.file, &pin, true)?;
            let metadata = held
                .file
                .metadata()
                .map_err(|e| ArenaError::Io(e.to_string()))?;
            check(
                metadata.is_file()
                    && (metadata.dev(), metadata.ino()) == original
                    && metadata.len() == held.pin.bytes,
                "/2 original retained FD identity/size changed",
            )?;
            let expected = match held.pin.path.as_str() {
                ROVENT_UTILITY_INTERVENTION_PUBLICATION_FILE => self.publication_raw.as_ref(),
                ROVENT_UTILITY_INTERVENTION_ACK_FILE => self.ack_raw.as_ref(),
                path => self.payloads.get(path),
            }
            .ok_or_else(|| invalid("/2 original FD lost retained payload"))?;
            check(raw == *expected, "/2 original retained raw bytes changed")?;
        }
        Ok(())
    }
    pub fn evidence(&self) -> Option<&RoventHistoricalUtilityInterventionEvidence> {
        self.evidence.as_ref()
    }
    pub fn publication(&self) -> Option<&RoventUtilityInterventionPublication> {
        self.publication.as_ref()
    }
    pub fn durable_ack(&self) -> Option<&RoventUtilityInterventionDurableAck> {
        self.ack.as_ref()
    }
    pub fn original_files(&self) -> impl Iterator<Item = (&RoventPublicationFile, &File)> {
        self.held.iter().map(|held| (&held.pin, &held.file))
    }
    pub fn read_accounting(&self) -> Value {
        self.budget.observation()
    }
    pub fn original_payloads(&self) -> &BTreeMap<String, Vec<u8>> {
        &self.payloads
    }
}

#[cfg(test)]
pub(in crate::rovent_collect::utility_intervention_owner) mod tests {
    use super::*;
    use std::{
        sync::atomic::{AtomicBool, AtomicU64, Ordering},
        time::{Duration, Instant},
    };
    static NEXT: AtomicU64 = AtomicU64::new(1);
    pub(in crate::rovent_collect::utility_intervention_owner) struct TestRoot {
        parent: PathBuf,
        root: PathBuf,
    }
    impl TestRoot {
        pub(in crate::rovent_collect::utility_intervention_owner) fn new() -> Self {
            let parent = std::env::temp_dir().join(format!(
                "rz-intervention-fd-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&parent).unwrap();
            let root = parent.join("utility-intervention-cohort");
            std::fs::create_dir(&root).unwrap();
            Self { parent, root }
        }
        pub(in crate::rovent_collect::utility_intervention_owner) fn directory(&self) -> &Path {
            &self.root
        }
    }
    impl Drop for TestRoot {
        fn drop(&mut self) {
            let _ =
                std::fs::remove_file(self.root.join(ROVENT_UTILITY_INTERVENTION_PUBLICATION_FILE));
            let _ = std::fs::remove_file(self.root.join(ROVENT_UTILITY_INTERVENTION_ACK_FILE));
            let _ = std::fs::remove_file(self.parent.join("outside.json"));
            let _ = std::fs::remove_dir(&self.root);
            let _ = std::fs::remove_dir(&self.parent);
        }
    }
    fn pin(path: &str, raw: &[u8]) -> RoventPublicationFile {
        RoventPublicationFile {
            path: path.into(),
            sha256: hex(Digest(Sha256::digest(raw).into())),
            bytes: raw.len() as u64,
        }
    }
    fn budget() -> Arc<RoventRecoveryReadBudget> {
        Arc::new(
            RoventRecoveryReadBudget::new(
                128 << 10,
                Instant::now() + Duration::from_secs(5),
                Arc::new(AtomicBool::new(false)),
            )
            .unwrap(),
        )
    }
    #[test]
    fn ready_publication_failure_retains_original_fd_and_charged_read_budget() {
        let root = TestRoot::new();
        let raw = br#"{"schema":"rz-rovent-utility-cohort-publication/1"}"#;
        std::fs::write(
            root.root.join(ROVENT_UTILITY_INTERVENTION_PUBLICATION_FILE),
            raw,
        )
        .unwrap();
        let rejected = RoventHistoricalUtilityInterventionFiles::open_retaining(
            root.root.clone(),
            pin(ROVENT_UTILITY_INTERVENTION_PUBLICATION_FILE, raw),
            pin(ROVENT_UTILITY_INTERVENTION_ACK_FILE, b"{}"),
            budget(),
        )
        .err()
        .expect("/1 is not admitted as /2");
        assert!(rejected.original.evidence().is_none());
        assert_eq!(rejected.original.original_files().count(), 1);
        assert_eq!(
            rejected.original.read_accounting()["charged_bytes"],
            raw.len() as u64
        );
        std::fs::remove_file(root.root.join(ROVENT_UTILITY_INTERVENTION_PUBLICATION_FILE)).unwrap();
        let (_, file) = rejected.original.original_files().next().unwrap();
        assert_eq!(file.metadata().unwrap().len(), raw.len() as u64);
    }
    #[test]
    fn historical_reader_does_not_follow_a_publication_symlink() {
        let root = TestRoot::new();
        let raw = b"outside original file";
        std::fs::write(root.parent.join("outside.json"), raw).unwrap();
        std::os::unix::fs::symlink(
            "../outside.json",
            root.root.join(ROVENT_UTILITY_INTERVENTION_PUBLICATION_FILE),
        )
        .unwrap();
        let rejected = RoventHistoricalUtilityInterventionFiles::open_retaining(
            root.root.clone(),
            pin(ROVENT_UTILITY_INTERVENTION_PUBLICATION_FILE, raw),
            pin(ROVENT_UTILITY_INTERVENTION_ACK_FILE, b"{}"),
            budget(),
        )
        .err()
        .expect("original /2 publication symlink refused");
        assert_eq!(rejected.original.original_files().count(), 0);
        assert_eq!(rejected.original.read_accounting()["charged_bytes"], 0);
    }
}
