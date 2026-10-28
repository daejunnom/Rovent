//! Declaration-only actor/source preflight for the original /2 Collection root.
//!
//! This keeps the same retained source bundle, root, manifest and control alive.
//! It does not start a teacher, authorize a go or label, prove actual build
//! provenance or worker liveness, or provide a physical completion receipt.
//! A later V2 session must still connect its original source custody to the
//! worker and handle revocation, shutdown, join and physical uncertainty.

use super::{ArenaError, RoventRegisteredTeacher, check, invalid};
use rz_contracts::Digest;
use rz_experiments::rovent_corpus_v2::RoventCorpusManifestV2;
use rz_uci::pals_adaptive::RoventCollectionRootAdmissionV2;
use rz_uci::rovent_bootstrap::{
    RoventCollectionPreloadSources, RoventCollectionTeacherSources, RoventStartupControl,
};
use std::{sync::Arc, time::Duration};

/// Retains the original bundle privately; its public fields cannot be replaced
/// through this object. No method here returns or submits the source bundle.
#[allow(dead_code)]
pub(crate) struct TeacherActorSourcePreflightV2 {
    admission: RoventCollectionRootAdmissionV2,
    preload: Arc<RoventCollectionPreloadSources>,
    startup_control: RoventStartupControl,
    root_control: RoventStartupControl,
    manifest: Arc<RoventCorpusManifestV2>,
    source: RoventCollectionTeacherSources,
    registered: RoventRegisteredTeacher,
    source_index: usize,
    collection_owner: u32,
    actor_digest: Digest,
    manifest_digest: String,
}

/// Compare one manifest actor against the exact source duplicated from the
/// root's original preload. The caller still checks the complete roster and
/// unique actor/index/owner assignments when assembling the actual session.
#[allow(dead_code)]
pub(crate) fn preflight_teacher_actor_source_v2(
    admission: &RoventCollectionRootAdmissionV2,
    startup_control: &RoventStartupControl,
    manifest: &Arc<RoventCorpusManifestV2>,
    actor_id: &str,
    source_index: usize,
    collection_owner: u32,
) -> Result<TeacherActorSourcePreflightV2, ArenaError> {
    manifest.validate()?;
    let manifest_digest = manifest.sha256()?;
    let root = admission.root_admission();
    let original = root
        .owner()
        .v2()
        .ok_or_else(|| invalid("teacher preflight requires the original /2 lock"))?;
    original.validate()?;
    let preload = admission.native_origin().collection_preload();
    let context = root.original_context();
    // Source admission was bounded by its own startup window. A later go has
    // a distinct cancellation owner and deadlines; do not extend the former
    // or use its expired deadline to reject this new root's bounded reads.
    let origin = root.runtime_clock().origin();
    let until = origin
        .checked_add(Duration::from_nanos(root.admission_deadline().at.0))
        .ok_or_else(|| invalid("teacher preflight root admission deadline overflow"))?;
    let cleanup_until = origin
        .checked_add(Duration::from_nanos(root.cleanup_deadline().at.0))
        .ok_or_else(|| invalid("teacher preflight root cleanup deadline overflow"))?;
    let root_control = RoventStartupControl {
        until,
        cleanup_until,
        cancellation: Arc::clone(&context.control.cancellation),
    };
    let source_declaration = rz_experiments::rovent_source_aggregate_sha256(
        &original.manifest.engine.source_commit,
        &original.manifest.engine.source_components,
    )?;
    check(
        collection_owner != 0
            && source_index < preload.teacher_count()
            && preload.matches_v2_origin(original, startup_control)
            && until <= cleanup_until
            && context
                .accepts()
                .map_err(|error| invalid(error.to_string()))?
            && manifest.source_commit == original.manifest.engine.source_commit
            && manifest.source_sha256 == source_declaration,
        "teacher preflight original root/preload/control or declared source mismatch",
    )?;
    let source = preload
        .teacher_sources(source_index, &root_control)
        .map_err(|error| invalid(error.to_string()))?;
    check(
        source.index == source_index
            && source
                .prepared
                .original_lock_owner_v2()
                .is_some_and(|lock| Arc::ptr_eq(lock, original)),
        "teacher preflight source does not belong to the original /2 lock/slot",
    )?;
    let registered = RoventRegisteredTeacher::from_prepared(
        &manifest.common_registration_projection(),
        actor_id,
        &source.prepared,
        &source.original_profile_bytes,
    )?;
    let actor_digest = registered.frozen_actor_digest()?;
    root_control
        .check()
        .map_err(|error| invalid(error.to_string()))?;
    check(
        context
            .accepts()
            .map_err(|error| invalid(error.to_string()))?,
        "teacher preflight original root revoked during source validation",
    )?;
    let preload = Arc::clone(preload);
    Ok(TeacherActorSourcePreflightV2 {
        admission: admission.clone(),
        preload,
        startup_control: startup_control.clone(),
        root_control,
        manifest: Arc::clone(manifest),
        source,
        registered,
        source_index,
        collection_owner,
        actor_digest,
        manifest_digest,
    })
}
