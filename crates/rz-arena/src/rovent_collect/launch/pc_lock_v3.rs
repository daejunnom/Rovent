//! Explicit /3 P-C Collection input preparation.
//!
//! These are original pinned declarations, not evidence that native work,
//! teachers, physical cleanup, or publication have completed.
use super::*;
use crate::rovent_collect::teacher::v3_session::TeacherRosterEntryV3;
use rz_experiments::rovent_corpus_lock_v3::record::RoventCorpusManifestLockV3;
use rz_experiments::{RoventActor, RoventInputLockV3, RoventPurpose};
use std::os::unix::fs::MetadataExt;

pub const ROVENT_COLLECTION_LOCK_V3_LAUNCH_PLAN: &str = "rz-rovent-collection-lock-v3-launch/1";
const PLAN_MAX_BYTES: u64 = 256 << 10;
const MANIFEST_MAX_BYTES: u64 = 16 << 20;

/// Separate, explicitly pinned Collection resource and roster declaration.
/// Worker limits, mailbox capacity and native role tickets are distinct.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventCollectionLockV3LaunchPlan {
    pub schema: String,
    pub manifest: RoventPublicationFile,
    pub resource_plan: RoventPublicationFile,
    pub metadata: RoventCollectionMetadata,
    pub teacher_sources: Vec<RoventTeacherSourceSelection>,
    pub teacher_slots: Vec<RoventCollectionTeacherSlot>,
    pub maximum_roots: u16,
    /// Maximum simultaneously owned collection root workers, including ones
    /// whose physical retirement remains unresolved.
    pub maximum_root_workers: u16,
    /// Separate teacher concurrency. The hook must reserve this capacity
    /// before starting actors and retain it until physical retirement.
    pub maximum_simultaneous_teacher_roots: u16,
    pub teacher_mailbox_capacity: u32,
    pub teacher_tickets_per_root: u32,
    pub maximum_records: u64,
}

/// The actual file descriptor and exact verified bytes remain alive with the
/// decoded object. Descriptor retention is not a claim that another process
/// cannot mutate the file; the product's private-input rules still apply.
struct OriginalPinnedFile {
    _file: cap_std::fs::File,
    bytes: Vec<u8>,
    pin: RoventPublicationFile,
}

impl OriginalPinnedFile {
    fn read(
        directory: &Dir,
        pin: &RoventPublicationFile,
        maximum: u64,
    ) -> Result<Self, ArenaError> {
        pin.validate()?;
        check(
            pin.path.starts_with("inputs/") && valid_relative(&pin.path) && pin.bytes <= maximum,
            "Collection /3 original private input path/size",
        )?;
        let mut options = cap_std::fs::OpenOptions::new();
        options.read(true).follow(FollowSymlinks::No);
        let mut file = directory.open_with(&pin.path, &options).map_err(io)?;
        let metadata = file.metadata().map_err(io)?;
        check(
            metadata.is_file() && metadata.len() == pin.bytes,
            "Collection /3 original file type/size",
        )?;
        let mut bytes = Vec::new();
        Read::take(&mut file, pin.bytes + 1)
            .read_to_end(&mut bytes)
            .map_err(io)?;
        check(
            bytes.len() as u64 == pin.bytes
                && hex(Digest(Sha256::digest(&bytes).into())) == pin.sha256,
            "Collection /3 original input bytes/SHA",
        )?;
        Ok(Self {
            _file: file,
            bytes,
            pin: pin.clone(),
        })
    }
}

/// Retains the original plan and manifest owners across resource admission.
pub struct PreparedPcCollectionInputsV3 {
    original_directory: std::fs::File,
    original_plan: OriginalPinnedFile,
    original_manifest: OriginalPinnedFile,
    plan: RoventCollectionLockV3LaunchPlan,
    manifest: Arc<RoventCorpusManifestLockV3>,
}

impl PreparedPcCollectionInputsV3 {
    pub fn open(directory: &Dir, plan_pin: &RoventPublicationFile) -> Result<Self, ArenaError> {
        let original_directory = directory.try_clone().map_err(io)?.into_std_file();
        let original_plan = OriginalPinnedFile::read(directory, plan_pin, PLAN_MAX_BYTES)?;
        let text = std::str::from_utf8(&original_plan.bytes)
            .map_err(|_| invalid("Collection /3 plan UTF8"))?;
        let plan: RoventCollectionLockV3LaunchPlan = rz_experiments::decode_json(text)?;
        check(
            plan.schema == ROVENT_COLLECTION_LOCK_V3_LAUNCH_PLAN,
            "Collection /3 explicit launch plan domain",
        )?;
        let original_manifest =
            OriginalPinnedFile::read(directory, &plan.manifest, MANIFEST_MAX_BYTES)?;
        let text = std::str::from_utf8(&original_manifest.bytes)
            .map_err(|_| invalid("Collection /3 manifest UTF8"))?;
        let manifest = Arc::new(RoventCorpusManifestLockV3::from_json(text)?);
        Ok(Self {
            original_directory,
            original_plan,
            original_manifest,
            plan,
            manifest,
        })
    }

    pub fn plan(&self) -> &RoventCollectionLockV3LaunchPlan {
        &self.plan
    }
    pub fn manifest(&self) -> &Arc<RoventCorpusManifestLockV3> {
        &self.manifest
    }
    pub fn original_plan_bytes(&self) -> &[u8] {
        &self.original_plan.bytes
    }
    pub fn original_manifest_bytes(&self) -> &[u8] {
        &self.original_manifest.bytes
    }
    pub fn original_plan_pin(&self) -> &RoventPublicationFile {
        &self.original_plan.pin
    }
    pub fn original_manifest_pin(&self) -> &RoventPublicationFile {
        &self.original_manifest.pin
    }

    /// Exact declared resource-plan pin. The original supervisor still opens
    /// and validates its own resource owner before calling lock admission.
    pub fn resource_plan_pin(
        &self,
    ) -> Result<crate::rovent_resource_scope::launch::RoventNativeResourcePlanPin, ArenaError> {
        let pin = &self.plan.resource_plan;
        pin.validate()?;
        check(
            pin.path.starts_with("inputs/") && valid_relative(&pin.path),
            "Collection /3 original resource-plan path",
        )?;
        Ok(
            crate::rovent_resource_scope::launch::RoventNativeResourcePlanPin {
                path: PathBuf::from(&pin.path),
                sha256: pin.sha256.clone(),
                bytes: pin.bytes,
            },
        )
    }

    /// Validate against the same /3 lock supplied by native resource
    /// admission. This registers no capability and starts no worker.
    pub fn validate_against(&self, lock: &RoventInputLockV3) -> Result<(), ArenaError> {
        self.manifest.validate(lock)?;
        self.plan.resource_plan.validate()?;
        check(
            lock.manifest.purpose == RoventPurpose::Collection
                && !lock.manifest.engine.mode.has_native_r()
                && self.plan.resource_plan.path.starts_with("inputs/")
                && valid_relative(&self.plan.resource_plan.path)
                && (1..=64).contains(&self.plan.maximum_roots)
                && (1..=self.plan.maximum_roots).contains(&self.plan.maximum_root_workers)
                && (1..=self.plan.maximum_root_workers)
                    .contains(&self.plan.maximum_simultaneous_teacher_roots)
                && self.plan.teacher_mailbox_capacity > 0
                && self.plan.teacher_mailbox_capacity
                    <= rz_experiments::rovent_corpus::MAX_CORPUS_ROWS as u32
                && self.plan.teacher_tickets_per_root > 0
                && self.plan.teacher_tickets_per_root
                    <= rz_experiments::rovent_corpus::MAX_CORPUS_ROWS as u32
                && (1..=rz_experiments::rovent_corpus::MAX_CORPUS_ROWS as u64)
                    .contains(&self.plan.maximum_records)
                && !self.plan.teacher_slots.is_empty()
                && self.plan.teacher_slots.len() == self.manifest.registration.teachers.len()
                && self.plan.teacher_sources.len() == self.manifest.registration.teachers.len(),
            "Collection /3 finite P-C plan or teacher coverage",
        )?;
        check(
            generation_identifier(&self.plan.metadata.game_id)
                && generation_identifier(&self.plan.metadata.opening_id)
                && generation_identifier(&self.plan.metadata.line_genealogy_id)
                && self
                    .manifest
                    .registration
                    .opening_splits
                    .contains_key(&self.plan.metadata.opening_id),
            "Collection /3 metadata/opening not registered",
        )?;
        let mut actors = BTreeSet::new();
        let mut indices = BTreeSet::new();
        let mut owners = BTreeSet::new();
        for slot in &self.plan.teacher_slots {
            check(
                self.manifest
                    .registration
                    .teachers
                    .contains_key(&slot.actor_id)
                    && actors.insert(&slot.actor_id)
                    && indices.insert(slot.source_index)
                    && slot.collection_owner > 0
                    && owners.insert(slot.collection_owner)
                    && slot.source_index < self.plan.teacher_sources.len()
                    && self.plan.teacher_sources[slot.source_index].index == slot.source_index,
                "Collection /3 actor/source/owner coverage",
            )?;
            let teacher = &self.manifest.registration.teachers[&slot.actor_id];
            let source = lock
                .manifest
                .engine
                .teachers
                .get(slot.source_index)
                .ok_or_else(|| invalid("Collection /3 teacher source not in original lock"))?;
            check(
                source.source_commit == teacher.source_commit
                    && source.binary.sha256 == teacher.binary_sha256
                    && source.binary.bytes == teacher.binary_bytes
                    && source.profile.sha256 == teacher.profile_sha256
                    && source.canonical_sha256 == teacher.canonical_sha256
                    && source.capability_sha256 == teacher.capability_sha256,
                "Collection /3 actor maps to a different original teacher source",
            )?;
        }
        let requested_threads = lock
            .manifest
            .engine
            .teachers
            .iter()
            .try_fold(0u32, |sum, teacher| {
                sum.checked_add(teacher.requested_threads)
            })
            .ok_or_else(|| invalid("Collection /3 teacher thread total overflow"))?;
        let simultaneous_threads = requested_threads
            .checked_mul(u32::from(self.plan.maximum_simultaneous_teacher_roots))
            .ok_or_else(|| invalid("Collection /3 simultaneous teacher threads overflow"))?;
        check(
            lock.manifest.resources.pools.iter().any(|pool| {
                pool.actors.contains(&RoventActor::Teacher)
                    && pool.max_threads >= simultaneous_threads
            }),
            "Collection /3 teacher workers exceed the original physical pool",
        )?;
        Ok(())
    }

    /// Compare the original open directory owner with the resource callback's
    /// original directory; equal path strings or equal hashes do not establish
    /// that both consumers accessed the same directory object.
    pub fn validate_original_boundary(
        &self,
        lock: &RoventInputLockV3,
        directory: &Dir,
        original_lock_file: &std::fs::File,
    ) -> Result<(), ArenaError> {
        let held = self.original_directory.metadata().map_err(io)?;
        let supplied = directory
            .try_clone()
            .map_err(io)?
            .into_std_file()
            .metadata()
            .map_err(io)?;
        let lock_file = original_lock_file.metadata().map_err(io)?;
        check(
            held.is_dir()
                && supplied.is_dir()
                && (held.dev(), held.ino()) == (supplied.dev(), supplied.ino())
                && lock_file.is_file(),
            "Collection /3 original resource boundary differs",
        )?;
        self.validate_against(lock)
    }

    pub(crate) fn roster(&self) -> Vec<TeacherRosterEntryV3> {
        self.plan
            .teacher_slots
            .iter()
            .map(|slot| TeacherRosterEntryV3 {
                actor_id: slot.actor_id.clone(),
                source_index: slot.source_index,
                collection_owner: slot.collection_owner,
            })
            .collect()
    }
}
