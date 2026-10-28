//! Original teacher source descriptors for explicit Collection preloading.
//!
//! This permission only admits native P/C source loading. It starts no teacher
//! process and grants no target, dispatch or completion authority. Each actual
//! root's collection owner consumes the retained files after Frontier admission.

use super::{PinnedBlob, open_blob, read_blob};
use crate::rovent_bootstrap::{RoventStartupControl, fail};
use crate::rovent_external_launch::{PreparedRoventExternalLaunch, RoventExternalSlot};
use cap_std::fs::Dir;
use rz_contracts::Digest;
use rz_eval::error::{BackendError, FailureKind as K};
use rz_experiments::{
    ArtifactRef, RoventEngineDeclaration, RoventInputLock, RoventInputLockV2, RoventPurpose,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeSet,
    fs::File,
    path::{Path, PathBuf},
    sync::Arc,
};

const PROFILE_MAX: usize = 256 << 10;
const BINARY_MAX: u64 = 256 << 20;
const ASSET_MAX: u64 = 1 << 30;

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventTeacherAssetSelection {
    pub purpose: String,
    pub artifact: ArtifactRef,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventTeacherSourceSelection {
    pub index: usize,
    /// A directory inside the same closed private input snapshot.
    pub working_directory_snapshot_path: String,
    pub assets: Vec<RoventTeacherAssetSelection>,
}

struct PinnedTeacherSource {
    profile: PinnedBlob,
    binary: PinnedBlob,
    cwd: File,
    assets: Vec<(String, PinnedBlob)>,
}

/// Preserve the actual parsed lock owner; never construct a legacy projection.
enum CollectionInputLock {
    V1(Arc<RoventInputLock>),
    V2(Arc<RoventInputLockV2>),
}
impl CollectionInputLock {
    fn validate(&self) -> Result<(), BackendError> {
        match self {
            Self::V1(lock) => lock.validate(),
            Self::V2(lock) => lock.validate(),
        }
        .map_err(|_| invalid("original Collection lock"))
    }
    fn purpose(&self) -> RoventPurpose {
        match self {
            Self::V1(lock) => lock.manifest.purpose,
            Self::V2(lock) => lock.manifest.purpose,
        }
    }
    fn engine(&self) -> &RoventEngineDeclaration {
        match self {
            Self::V1(lock) => &lock.manifest.engine,
            Self::V2(lock) => &lock.manifest.engine,
        }
    }
    fn provenance(&self) -> &[ArtifactRef] {
        match self {
            Self::V1(lock) => &lock.manifest.provenance,
            Self::V2(lock) => &lock.manifest.provenance,
        }
    }
    fn canonical_sha256(&self) -> &str {
        match self {
            Self::V1(lock) => &lock.canonical_sha256,
            Self::V2(lock) => &lock.canonical_sha256,
        }
    }
    fn teacher_launch(
        &self,
        index: usize,
        bytes: &[u8],
    ) -> Result<PreparedRoventExternalLaunch, BackendError> {
        let slot = RoventExternalSlot::Teacher { index };
        match self {
            Self::V1(lock) => PreparedRoventExternalLaunch::from_lock(lock, slot, bytes),
            Self::V2(lock) => PreparedRoventExternalLaunch::from_lock_v2(lock, slot, bytes),
        }
        .map_err(|_| invalid("original teacher profile/capability declaration"))
    }
}

/// Only `open` can create this source permission from the full original lock
/// and its exact retained profile, executable, cwd and separate-asset files.
pub struct RoventCollectionPreloadSources {
    lock: CollectionInputLock,
    /// /2 retains the original source-admission control for later owner joins.
    /// /1 keeps its existing behavior and does not retain an extra control.
    startup_control_v2: Option<RoventStartupControl>,
    // This is the path used for profile program/cwd checks, not proof that a
    // later pathname lookup resolves to the held directory capability.
    private_root_path_v2: Option<PathBuf>,
    _original_root: Dir,
    teachers: Vec<PinnedTeacherSource>,
}

/// Original descriptors transferred to an upper collection session owner.
/// It still needs the actual per-go root/Frontier and real teacher startup.
pub struct RoventCollectionTeacherSources {
    pub index: usize,
    pub prepared: PreparedRoventExternalLaunch,
    pub original_profile_bytes: Vec<u8>,
    pub executable: File,
    pub working_directory: File,
    pub assets: Vec<(String, ArtifactRef, File)>,
    original: Arc<RoventCollectionPreloadSources>,
}
/// Retain this after the worker consumes its executable/profile descriptors.
/// It is source custody only and cannot submit or validate a teacher result.
pub struct RoventCollectionTeacherSourceCustody {
    original: Arc<RoventCollectionPreloadSources>,
    assets: Vec<(String, ArtifactRef, File)>,
}
impl RoventCollectionTeacherSourceCustody {
    /// Original source owner identity; equal digests alone do not join a
    /// teacher worker to the native source that admitted Collection.
    pub fn matches_preload(&self, preload: &Arc<RoventCollectionPreloadSources>) -> bool {
        Arc::ptr_eq(&self.original, preload)
    }
    pub fn original_input_lock_digest(&self) -> Digest {
        self.original.input_lock_digest()
    }
    pub fn separate_assets(&self) -> &[(String, ArtifactRef, File)] {
        &self.assets
    }
}
impl RoventCollectionTeacherSources {
    pub fn original_input_lock_digest(&self) -> Digest {
        self.original.input_lock_digest()
    }
    pub fn into_parts(
        self,
    ) -> (
        usize,
        PreparedRoventExternalLaunch,
        Vec<u8>,
        File,
        File,
        RoventCollectionTeacherSourceCustody,
    ) {
        (
            self.index,
            self.prepared,
            self.original_profile_bytes,
            self.executable,
            self.working_directory,
            RoventCollectionTeacherSourceCustody {
                original: self.original,
                assets: self.assets,
            },
        )
    }
}

impl RoventCollectionPreloadSources {
    pub fn open(
        lock: Arc<RoventInputLock>,
        root: Dir,
        private_root_path: &Path,
        selections: Vec<RoventTeacherSourceSelection>,
        control: &RoventStartupControl,
    ) -> Result<Self, BackendError> {
        Self::open_original(
            CollectionInputLock::V1(lock),
            root,
            private_root_path,
            selections,
            control,
        )
    }

    /// The actual /2 Arc and its held descriptors remain together. This does
    /// not start a teacher or create a root/target authority.
    pub fn open_v2(
        lock: Arc<RoventInputLockV2>,
        root: Dir,
        private_root_path: &Path,
        selections: Vec<RoventTeacherSourceSelection>,
        control: &RoventStartupControl,
    ) -> Result<Self, BackendError> {
        Self::open_original(
            CollectionInputLock::V2(lock),
            root,
            private_root_path,
            selections,
            control,
        )
    }

    fn open_original(
        lock: CollectionInputLock,
        root: Dir,
        private_root_path: &Path,
        selections: Vec<RoventTeacherSourceSelection>,
        control: &RoventStartupControl,
    ) -> Result<Self, BackendError> {
        control.check()?;
        lock.validate()?;
        let startup_control_v2 =
            matches!(&lock, CollectionInputLock::V2(_)).then(|| control.clone());
        let private_root_path_v2 =
            matches!(&lock, CollectionInputLock::V2(_)).then(|| private_root_path.to_owned());
        if lock.purpose() != RoventPurpose::Collection
            || lock.engine().teachers.is_empty()
            || lock.engine().teachers.len() > 16
            || selections.len() != lock.engine().teachers.len()
            || !private_root_path.is_absolute()
        {
            return Err(invalid(
                "explicit Collection teacher source cardinality/purpose",
            ));
        }
        let mut teachers = Vec::new();
        teachers.try_reserve_exact(selections.len()).map_err(|_| {
            fail(
                K::ResourceExhausted,
                "teacher source roster allocation failed",
            )
        })?;
        for (index, selection) in selections.into_iter().enumerate() {
            let pin = &lock.engine().teachers[index];
            if selection.index != index
                || !rz_experiments::pals_private_snapshot_path_v5(&pin.profile.path)
                || !rz_experiments::pals_private_snapshot_path_v5(&pin.binary.path)
                || !rz_experiments::pals_private_snapshot_path_v5(
                    &selection.working_directory_snapshot_path,
                )
                || selection.assets.len() > 32
            {
                return Err(invalid(
                    "teacher sources must cover original slots in exact order",
                ));
            }
            let profile = open_blob(&root, pin.profile.clone(), PROFILE_MAX as u64)?;
            let bytes = read_blob(&profile, PROFILE_MAX)?;
            control.check()?;
            let launch = lock.teacher_launch(index, &bytes)?;
            let declaration = launch.profile();
            if Path::new(&declaration.program) != private_root_path.join(&pin.binary.path)
                || Path::new(&declaration.working_directory)
                    != private_root_path.join(&selection.working_directory_snapshot_path)
            {
                return Err(invalid(
                    "teacher program/cwd differs from the original snapshot mapping",
                ));
            }
            let binary = open_blob(&root, pin.binary.clone(), BINARY_MAX)?;
            control.check()?;
            let cwd = root
                .open_dir(&selection.working_directory_snapshot_path)
                .map_err(|_| invalid("original teacher working-directory descriptor"))?
                .into_std_file();
            if !cwd.metadata().is_ok_and(|metadata| metadata.is_dir()) {
                return Err(invalid(
                    "teacher working-directory descriptor is not a directory",
                ));
            }
            let expected = &declaration.identity.assets;
            if selection.assets.len() != expected.len() {
                return Err(invalid("teacher separate asset roster coverage differs"));
            }
            let mut seen = BTreeSet::new();
            let mut assets = Vec::new();
            for (selected, expected) in selection.assets.into_iter().zip(expected) {
                if selected.purpose != expected.purpose
                    || selected.artifact.sha256 != expected.sha256
                    || !rz_experiments::pals_private_snapshot_path_v5(&selected.artifact.path)
                    || !lock.provenance().contains(&selected.artifact)
                    || !seen.insert((selected.purpose.clone(), selected.artifact.path.clone()))
                {
                    return Err(invalid(
                        "teacher separate asset purpose/hash/source roster differs",
                    ));
                }
                assets.push((
                    selected.purpose,
                    open_blob(&root, selected.artifact, ASSET_MAX)?,
                ));
                control.check()?;
            }
            teachers.push(PinnedTeacherSource {
                profile,
                binary,
                cwd,
                assets,
            });
        }
        control.check()?;
        Ok(Self {
            lock,
            startup_control_v2,
            private_root_path_v2,
            _original_root: root,
            teachers,
        })
    }

    pub fn input_lock_digest(&self) -> Digest {
        // The original lock was fully validated before this source owner arose.
        let bytes = rz_eval::asset::parse_sha256(self.lock.canonical_sha256())
            .expect("validated original Collection lock digest");
        Digest(bytes)
    }
    pub fn matches_input_lock(&self, lock: &RoventInputLock) -> bool {
        matches!(&self.lock, CollectionInputLock::V1(original) if original.as_ref() == lock)
    }
    /// /2 custody is bound to the exact original Arc, not merely equal fields.
    pub fn matches_input_lock_v2(&self, lock: &Arc<RoventInputLockV2>) -> bool {
        matches!(&self.lock, CollectionInputLock::V2(original) if Arc::ptr_eq(original, lock))
    }
    /// Check the original /2 lock and the exact source-admission cancellation
    /// allocation and deadlines. This observation itself grants no authority.
    pub fn matches_v2_origin(
        &self,
        lock: &Arc<RoventInputLockV2>,
        control: &RoventStartupControl,
    ) -> bool {
        self.matches_input_lock_v2(lock)
            && self.startup_control_v2.as_ref().is_some_and(|original| {
                Arc::ptr_eq(&original.cancellation, &control.cancellation)
                    && original.until == control.until
                    && original.cleanup_until == control.cleanup_until
            })
    }
    /// Compare held directory capabilities for an eventual same-owner join.
    /// Directory identity alone is not a process or root admission witness.
    pub fn matches_original_root_v2(&self, root: &Dir) -> Result<bool, BackendError> {
        if !matches!(&self.lock, CollectionInputLock::V2(_)) {
            return Ok(false);
        }
        super::same_directory_identity(&self._original_root, root)
    }
    /// Also require the exact path used in the original profile mapping.
    /// The retained directory identity must be checked independently.
    pub fn matches_original_root_path_v2(&self, private_root_path: &Path) -> bool {
        self.private_root_path_v2.as_deref() == Some(private_root_path)
    }
    pub fn teacher_count(&self) -> usize {
        self.teachers.len()
    }

    /// Duplicate only the original retained descriptors for a root's unique
    /// teacher owner. No pathname is reopened and no startup is represented.
    pub fn teacher_sources(
        self: &Arc<Self>,
        index: usize,
        control: &RoventStartupControl,
    ) -> Result<RoventCollectionTeacherSources, BackendError> {
        control.check()?;
        let source = self
            .teachers
            .get(index)
            .ok_or_else(|| invalid("teacher source slot"))?;
        let original_profile_bytes = read_blob(&source.profile, PROFILE_MAX)?;
        let prepared = self
            .lock
            .teacher_launch(index, &original_profile_bytes)
            .map_err(|_| invalid("retained original teacher profile changed"))?;
        let duplicate = |file: &File| {
            file.try_clone()
                .map_err(|_| fail(K::Io, "original teacher descriptor duplication failed"))
        };
        let executable = duplicate(&source.binary.file)?;
        let working_directory = duplicate(&source.cwd)?;
        let mut assets = Vec::new();
        for (purpose, blob) in &source.assets {
            assets.push((
                purpose.clone(),
                blob.declaration.clone(),
                duplicate(&blob.file)?,
            ));
        }
        control.check()?;
        Ok(RoventCollectionTeacherSources {
            index,
            prepared,
            original_profile_bytes,
            executable,
            working_directory,
            assets,
            original: Arc::clone(self),
        })
    }
}
fn invalid(detail: &'static str) -> BackendError {
    fail(K::IdentityMismatch, detail)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rz_contracts::rovent::ExecutionMode;
    use rz_experiments::{
        ROVENT_EXECUTION_V2_DOMAIN, RoventExecutionManifest, RoventExecutionManifestV2,
    };
    use serde_json::json;
    use std::sync::atomic::AtomicBool;
    use std::time::{Duration, Instant};

    #[test]
    fn v2_original_lock_and_source_control_cannot_be_replaced_by_equal_values() {
        let v1_json = crate::rovent_bootstrap::tests::wire(ExecutionMode::CpuPc);
        let v1 = RoventExecutionManifest::from_json(&v1_json.to_string())
            .unwrap()
            .lock()
            .unwrap();
        let mut v2_json = v1_json;
        v2_json["domain"] = json!(ROVENT_EXECUTION_V2_DOMAIN);
        v2_json["startup_document"] = v2_json["engine"]["binary"].clone();
        v2_json["startup_configuration_sha256"] = json!("4".repeat(64));
        v2_json["startup_search_configuration_sha256"] = json!("1".repeat(64));
        v2_json["own_r_root_tie"] = json!({"kind":"not_applicable", "reason":"pc_only"});
        let lock = Arc::new(
            RoventExecutionManifestV2::from_json(&v2_json.to_string())
                .unwrap()
                .lock()
                .unwrap(),
        );
        let now = Instant::now();
        let control = RoventStartupControl {
            until: now + Duration::from_secs(10),
            cleanup_until: now + Duration::from_secs(20),
            cancellation: Arc::new(AtomicBool::new(false)),
        };
        let root = Dir::open_ambient_dir(".", cap_std::ambient_authority()).unwrap();
        let owner = RoventCollectionPreloadSources {
            lock: CollectionInputLock::V2(Arc::clone(&lock)),
            startup_control_v2: Some(control.clone()),
            private_root_path_v2: Some(PathBuf::from("/original/snapshot")),
            _original_root: root,
            teachers: Vec::new(),
        };
        assert!(owner.matches_v2_origin(&lock, &control.clone()));
        assert!(!owner.matches_v2_origin(&Arc::new((*lock).clone()), &control));
        assert!(!owner.matches_input_lock(&v1));
        assert!(owner.matches_original_root_path_v2(Path::new("/original/snapshot")));
        assert!(!owner.matches_original_root_path_v2(Path::new("/different/snapshot")));
        let foreign = RoventStartupControl {
            cancellation: Arc::new(AtomicBool::new(false)),
            ..control.clone()
        };
        assert!(!owner.matches_v2_origin(&lock, &foreign));
        let expired = RoventStartupControl {
            until: control.until + Duration::from_millis(1),
            ..control.clone()
        };
        assert!(!owner.matches_v2_origin(&lock, &expired));
        let changed_cleanup = RoventStartupControl {
            cleanup_until: control.cleanup_until + Duration::from_millis(1),
            ..control.clone()
        };
        assert!(!owner.matches_v2_origin(&lock, &changed_cleanup));
        #[cfg(target_os = "linux")]
        {
            let same_root = Dir::open_ambient_dir(".", cap_std::ambient_authority()).unwrap();
            assert!(owner.matches_original_root_v2(&same_root).unwrap());
            let other_root = Dir::open_ambient_dir("..", cap_std::ambient_authority()).unwrap();
            assert!(!owner.matches_original_root_v2(&other_root).unwrap());
        }
        assert_eq!(
            owner.input_lock_digest(),
            Digest(rz_eval::asset::parse_sha256(&lock.canonical_sha256).unwrap())
        );
        assert!(matches!(
            PreparedRoventExternalLaunch::from_lock_v2(
                &lock,
                RoventExternalSlot::Teacher { index: 0 },
                b"{}"
            ),
            Err(rz_search::cpu_checker::CheckerError::Unsupported(_))
        ));
    }
}
