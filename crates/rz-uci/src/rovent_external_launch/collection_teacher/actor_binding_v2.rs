//! Declared corpus actor bound to the exact /2 root's original teacher source.
//!
//! This is source/recipe admission only. A caller-supplied manifest `Arc` is
//! not evidence of a registered manifest FD, actual worker execution, label
//! authority, physical completion, or complete roster coverage.

use super::*;
use crate::pals_adaptive::RoventCollectionRootAdmissionV2;
use crate::pals_checker_profile::rovent::RoventExternalProfile;
use crate::rovent_bootstrap::{RoventCollectionTeacherSources, RoventStartupControl};
use rz_contracts::Digest;
use rz_experiments::pals_data_v3::generation_digest;
use rz_experiments::rovent_corpus::RoventTeacherPin;
use rz_experiments::rovent_corpus_v2::RoventCorpusManifestV2;
use rz_search::cpu_checker::CheckerError;
use rz_search::external_cpu::{ExternalEngineProfile, ExternalEnginePurpose};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

const FROZEN_ACTOR_DOMAIN: &str = "rz-rovent-frozen-teacher-actor/1";

/// Root contention/poisoning is not a revoked root. Preserve the original
/// structured cause and stage for source and manifest failures as well.
#[derive(Debug)]
pub enum TeacherActorBindingErrorV2 {
    Invalid(&'static str),
    RootObservation(rz_contracts::ContractError),
    Source {
        stage: &'static str,
        error: rz_eval::error::BackendError,
    },
    Manifest {
        stage: &'static str,
        error: rz_experiments::ManifestError,
    },
    Pin(CheckerError),
}
impl std::fmt::Display for TeacherActorBindingErrorV2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(detail) => write!(f, "teacher actor binding: {detail}"),
            Self::RootObservation(error) => write!(f, "teacher root observation: {error}"),
            Self::Source { stage, error } => write!(f, "teacher source {stage}: {error}"),
            Self::Manifest { stage, error } => write!(f, "teacher manifest {stage}: {error}"),
            Self::Pin(error) => write!(f, "teacher pin: {error}"),
        }
    }
}
impl std::error::Error for TeacherActorBindingErrorV2 {}

/// Reuse the same pin/profile comparison for original V1 registration and V2
/// binding. This validates declarations only; it cannot issue execution rights.
pub fn verify_collection_teacher_pin(
    pin: &RoventTeacherPin,
    launch: &PreparedRoventExternalLaunch,
    original_profile_bytes: &[u8],
) -> Result<(), CheckerError> {
    let loaded =
        RoventExternalProfile::from_verified_bytes(original_profile_bytes, &pin.profile_sha256)?;
    let profile = loaded.profile();
    let family = match profile.engine {
        ExternalEngineProfile::Stockfish19EmbeddedNnue => "stockfish",
        ExternalEngineProfile::Reckless090EmbeddedNnue => "reckless",
    };
    let assets = profile
        .identity
        .assets
        .iter()
        .map(|asset| (asset.purpose.clone(), asset.sha256.clone()))
        .collect::<BTreeMap<_, _>>();
    if profile != launch.profile()
        || profile.purpose != ExternalEnginePurpose::Teacher
        || pin.engine != family
        || profile.engine.source_commit() != pin.source_commit
        || profile.identity.binary_sha256 != pin.binary_sha256
        || launch.binary().sha256 != pin.binary_sha256
        || launch.binary().bytes != pin.binary_bytes
        || loaded.registration().canonical_sha256 != pin.canonical_sha256
        || launch.canonical_sha256() != pin.canonical_sha256
        || launch.capability_sha256() != pin.capability_sha256
        || profile.identity.options != pin.options
        || assets.len() != profile.identity.assets.len()
        || assets != pin.assets
    {
        return Err(invalid(
            "teacher recipe/profile/source/binary/options/assets do not match retained launch",
        ));
    }
    Ok(())
}

/// A declaration-bound actor retaining the real /2 root and source bundle.
/// Fields are private so an arbitrary digest cannot manufacture this owner.
pub struct BoundCollectionTeacherActorV2 {
    admission: RoventCollectionRootAdmissionV2,
    manifest: Arc<RoventCorpusManifestV2>,
    source: RoventCollectionTeacherSources,
    root_control: RoventStartupControl,
    actor_id: String,
    actor_digest: Digest,
    manifest_sha256: String,
    source_index: usize,
    collection_owner: u32,
}

impl BoundCollectionTeacherActorV2 {
    pub fn actor_id(&self) -> &str {
        &self.actor_id
    }
    pub fn actor_digest(&self) -> Digest {
        self.actor_digest
    }
    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }
    pub fn source_index(&self) -> usize {
        self.source_index
    }
    pub fn collection_owner(&self) -> u32 {
        self.collection_owner
    }

    pub(super) fn admission(&self) -> &RoventCollectionRootAdmissionV2 {
        &self.admission
    }
    pub(super) fn manifest(&self) -> &Arc<RoventCorpusManifestV2> {
        &self.manifest
    }
    pub(super) fn source(&self) -> &RoventCollectionTeacherSources {
        &self.source
    }
    pub(super) fn root_control(&self) -> &RoventStartupControl {
        &self.root_control
    }

    /// The collection teacher module consumes this exact bundle, not a new
    /// descriptor set reopened from an index or a reconstructed capability.
    pub(super) fn into_parts(self) -> BoundCollectionTeacherActorPartsV2 {
        BoundCollectionTeacherActorPartsV2 {
            admission: self.admission,
            manifest: self.manifest,
            source: self.source,
            root_control: self.root_control,
            actor_id: self.actor_id,
            actor_digest: self.actor_digest,
            manifest_sha256: self.manifest_sha256,
            source_index: self.source_index,
            collection_owner: self.collection_owner,
        }
    }

    /// Restore the very same retained bundle on a failure before worker
    /// creation. This is private to the collection teacher implementation.
    pub(super) fn from_parts(parts: BoundCollectionTeacherActorPartsV2) -> Self {
        Self {
            admission: parts.admission,
            manifest: parts.manifest,
            source: parts.source,
            root_control: parts.root_control,
            actor_id: parts.actor_id,
            actor_digest: parts.actor_digest,
            manifest_sha256: parts.manifest_sha256,
            source_index: parts.source_index,
            collection_owner: parts.collection_owner,
        }
    }
}

pub(super) struct BoundCollectionTeacherActorPartsV2 {
    pub(super) admission: RoventCollectionRootAdmissionV2,
    pub(super) manifest: Arc<RoventCorpusManifestV2>,
    pub(super) source: RoventCollectionTeacherSources,
    pub(super) root_control: RoventStartupControl,
    pub(super) actor_id: String,
    pub(super) actor_digest: Digest,
    pub(super) manifest_sha256: String,
    pub(super) source_index: usize,
    pub(super) collection_owner: u32,
}

/// Binds one declared actor to a retained original source; caller separately
/// verifies complete roster uniqueness and the registered manifest source.
pub fn bind_collection_teacher_actor_v2(
    admission: &RoventCollectionRootAdmissionV2,
    startup_control: &RoventStartupControl,
    manifest: &Arc<RoventCorpusManifestV2>,
    actor_id: &str,
    source_index: usize,
    collection_owner: u32,
) -> Result<BoundCollectionTeacherActorV2, TeacherActorBindingErrorV2> {
    manifest
        .validate()
        .map_err(|error| TeacherActorBindingErrorV2::Manifest {
            stage: "validate",
            error,
        })?;
    let manifest_sha256 =
        manifest
            .sha256()
            .map_err(|error| TeacherActorBindingErrorV2::Manifest {
                stage: "digest",
                error,
            })?;
    let pin = manifest
        .teachers
        .get(actor_id)
        .ok_or(TeacherActorBindingErrorV2::Invalid("teacher actor absent"))?;
    let root = admission.root_admission();
    let lock = root
        .owner()
        .v2()
        .ok_or(TeacherActorBindingErrorV2::Invalid(
            "teacher original /2 lock absent",
        ))?;
    lock.validate()
        .map_err(|error| TeacherActorBindingErrorV2::Manifest {
            stage: "input-lock",
            error,
        })?;
    let preload = admission.native_origin().collection_preload();
    let source_preload = admission
        .native_origin()
        .sources()
        .verify_collection_v2_origin()
        .map_err(|error| TeacherActorBindingErrorV2::Source {
            stage: "original-preload",
            error,
        })?;
    let context = root.original_context();
    let origin = root.runtime_clock().origin();
    let until = origin
        .checked_add(Duration::from_nanos(root.admission_deadline().at.0))
        .ok_or(TeacherActorBindingErrorV2::Invalid(
            "teacher root deadline overflow",
        ))?;
    let cleanup_until = origin
        .checked_add(Duration::from_nanos(root.cleanup_deadline().at.0))
        .ok_or(TeacherActorBindingErrorV2::Invalid(
            "teacher root cleanup overflow",
        ))?;
    let root_control = RoventStartupControl {
        until,
        cleanup_until,
        cancellation: Arc::clone(&context.control.cancellation),
    };
    let source_declaration = rz_experiments::rovent_source_aggregate_sha256(
        &lock.manifest.engine.source_commit,
        &lock.manifest.engine.source_components,
    )
    .map_err(|error| TeacherActorBindingErrorV2::Manifest {
        stage: "source-aggregate",
        error,
    })?;
    let root_accepts = context
        .try_accepts()
        .map_err(TeacherActorBindingErrorV2::RootObservation)?;
    if collection_owner == 0
        || source_index >= preload.teacher_count()
        || !Arc::ptr_eq(preload, source_preload)
        || !preload.matches_v2_origin(lock, startup_control)
        || until > cleanup_until
        || !root_accepts
        || manifest.source_commit != lock.manifest.engine.source_commit
        || manifest.source_sha256 != source_declaration
    {
        return Err(TeacherActorBindingErrorV2::Invalid(
            "teacher original root/preload/control or source declaration mismatch",
        ));
    }
    let source = preload
        .teacher_sources(source_index, &root_control)
        .map_err(|error| TeacherActorBindingErrorV2::Source {
            stage: "teacher-source",
            error,
        })?;
    if source.index != source_index
        || !source
            .prepared
            .original_lock_owner_v2()
            .is_some_and(|owner| Arc::ptr_eq(owner, lock))
    {
        return Err(TeacherActorBindingErrorV2::Invalid(
            "teacher source lock/slot mismatch",
        ));
    }
    verify_collection_teacher_pin(pin, &source.prepared, &source.original_profile_bytes)
        .map_err(TeacherActorBindingErrorV2::Pin)?;
    let actor_hex = generation_digest(
        FROZEN_ACTOR_DOMAIN,
        &serde_json::json!({
            "actor_id": actor_id,
            "teacher_pin": pin,
        }),
    )
    .map_err(|error| TeacherActorBindingErrorV2::Manifest {
        stage: "actor-digest",
        error,
    })?;
    let actor_digest = Digest(rz_eval::asset::parse_sha256(&actor_hex).map_err(|error| {
        TeacherActorBindingErrorV2::Source {
            stage: "actor-digest-syntax",
            error,
        }
    })?);
    root_control
        .check()
        .map_err(|error| TeacherActorBindingErrorV2::Source {
            stage: "root-control",
            error,
        })?;
    if !context
        .try_accepts()
        .map_err(TeacherActorBindingErrorV2::RootObservation)?
    {
        return Err(TeacherActorBindingErrorV2::Invalid(
            "teacher original root no longer accepts",
        ));
    }
    Ok(BoundCollectionTeacherActorV2 {
        admission: admission.clone(),
        manifest: Arc::clone(manifest),
        source,
        root_control,
        actor_id: actor_id.to_owned(),
        actor_digest,
        manifest_sha256,
        source_index,
        collection_owner,
    })
}
