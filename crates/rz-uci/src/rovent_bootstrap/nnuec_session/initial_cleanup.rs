//! Bounded physical cleanup of the original, pre-admission NNUEC owners.
//!
//! Actual engines follow the cleanup-only same-actor transfer. Only the
//! preparing root's typed retirement API may return its same semantic owner,
//! after actual release/cache witnesses and matching external pin release.
//!
//! Every step crosses at most one non-waiting actor/worker boundary. The caller
//! supplies its original cleanup deadline and retains Pending on expiry. Both
//! Pending and Retained quarantine their custody if accidentally dropped, so
//! CpuWorker/PreparationActor's blocking destructors cannot run in that path.

use super::super::nnuec_preparing::{
    InitialNnuecCleanup, InitialNnuecCleanupOwners, InitialNnuecFailure, InitialNnuecPhase,
};
use rz_search::cpu_value::nnuec::v2::retirement::{NnuecCacheInvalidation, NnuecCacheReleaseAck};
use rz_search::cpu_value::nnuec::v2::review::{
    Completion, PreparationActor, PreparationError, PreparationObservation, Provenance,
};
use rz_search::cpu_worker::{
    CpuWorkerError, CpuWorkerShutdown, CpuWorkerThreadAffinityShutdown, MaterializationCompletion,
    NnuecAuthorityRevocation, NnuecEngineReleaseAck, NnuecEngineRetirementCompletion,
    NnuecEngineRetirementId, NnuecMaterializationReleaseAck, NnuecMaterializationReleaseCompletion,
    NnuecMaterializationReleaseId, RecoveredNnuecEngine,
};
use rz_search::pals::engine::adaptive_port::AdaptivePortModel;
use rz_search::pals::engine::preparing_root::{
    InitialPreparingRetirementRejection, InitialResearcherRelease, InitialResearcherReleaseReceipt,
    ResearcherPreparingError, RetiredPreparingRoot,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NnuecInitialCleanupPhase {
    CancelRoot,
    StopStartingWorker,
    StopReadyWorker,
    InvalidateCache,
    DrainReview,
    DrainMaterialization,
    ClaimMaterialization,
    JoinStartingWorker,
    RecoverStartingEngine,
    JoinReadyWorker,
    RecoverReadyEngine,
    SubmitMaterializationRelease,
    AwaitMaterializationRelease,
    SubmitEngineRetirement,
    AwaitEngineRetirement,
    AwaitCacheRelease,
    SelectSemanticRelease,
    ReleaseExternalPins,
    RetireSemanticRoot,
    StopActor,
    JoinActor,
    Retain,
}

#[derive(Debug)]
pub enum NnuecInitialCleanupIssue {
    Preparation(PreparationError),
    Worker(CpuWorkerError),
    PreparingRoot(ResearcherPreparingError),
    Invalid(&'static str),
}

/// Actual collected receipts. Missing observations never mean completion.
/// All retained terminals keep their original provenance and unique owners.
#[derive(Default)]
pub struct NnuecInitialCleanupObservations {
    pub root_cancel_requested: bool,
    pub starting_shutdown_requested: bool,
    pub ready_shutdown_requested: bool,
    pub actor_shutdown_requested: bool,
    pub drained_review: Option<Completion>,
    pub review_matches_original: Option<bool>,
    pub review_terminal_provenance: Option<Provenance>,
    pub review_terminal_observation: Option<PreparationObservation>,
    pub review_terminal_result: Option<Result<(), PreparationError>>,
    pub drained_materialization: Option<MaterializationCompletion>,
    pub materialization_matches_original: Option<bool>,
    /// The exact matching successful terminal moved its original handoff into
    /// owners before claim/release. A mismatching terminal stays whole above.
    pub materialization_handoff_provenance: Option<Provenance>,
    pub materialization_terminal_error: Option<CpuWorkerError>,
    pub starting_shutdown: Option<CpuWorkerShutdown>,
    pub ready_shutdown: Option<CpuWorkerShutdown>,
    pub recovered_engine: Option<RecoveredNnuecEngine>,
    pub retirement_request: Option<NnuecEngineRetirementId>,
    pub unmatched_retirement: Option<NnuecEngineRetirementCompletion>,
    pub engine_release: Option<NnuecEngineReleaseAck>,
    pub engine_release_matches_original: Option<bool>,
    pub materialization_release_request: Option<NnuecMaterializationReleaseId>,
    pub unmatched_materialization_release: Option<NnuecMaterializationReleaseCompletion>,
    pub materialization_release: Option<NnuecMaterializationReleaseAck>,
    pub materialization_release_matches_original: Option<bool>,
    pub cache_invalidation: Option<NnuecCacheInvalidation>,
    pub cache_release: Option<NnuecCacheReleaseAck>,
    pub cache_clears_released_engine: Option<bool>,
    pub cache_clears_released_materialization: Option<bool>,
    pub actor_join: Option<Result<(), PreparationError>>,
    pub semantic_rejection: Option<InitialPreparingRetirementRejection<AdaptivePortModel>>,
    pub issues: Vec<NnuecInitialCleanupIssue>,
}

struct Custody {
    original_phase: InitialNnuecPhase,
    original_failure: Option<InitialNnuecFailure>,
    owners: InitialNnuecCleanupOwners,
    observations: NnuecInitialCleanupObservations,
    selected_release: Option<InitialResearcherRelease>,
}

/// Physical observations plus the exact unreleased semantic/source owners.
/// This variant never asserts semantic retirement or a reusable actor.
#[must_use = "physical cleanup observations do not return the semantic owner"]
pub struct NnuecInitialCleanupRetained {
    custody: Option<Box<Custody>>,
}
impl Drop for NnuecInitialCleanupRetained {
    fn drop(&mut self) {
        // No successful semantic release attests these pins. Any failed physical
        // transfer can also retain its actual CpuEngine/TT in this custody.
        if let Some(custody) = self.custody.take() {
            std::mem::forget(custody);
        }
    }
}

#[must_use = "retain pending or terminal actual cleanup custody"]
// The caller holds one cleanup step at a time. Ready moves the same retired
// owner, actor and release observations; it does not grow a pending queue.
#[allow(clippy::large_enum_variant)]
pub enum NnuecInitialCleanupStep {
    Pending(NnuecInitialCleanup),
    Ready(NnuecInitialCleaned),
    Retained(NnuecInitialCleanupRetained),
}

/// The original semantic owner was returned by its actual typed retirement
/// API. The same preparation actor remains alive for the next root.
#[must_use = "transfer the actual retired owner and same actor into session custody"]
pub struct NnuecInitialCleaned {
    pub retired: RetiredPreparingRoot<AdaptivePortModel>,
    pub actor: PreparationActor,
    pub original_phase: InitialNnuecPhase,
    pub original_failure: Option<InitialNnuecFailure>,
    pub release: Option<InitialResearcherReleaseReceipt>,
    pub cache_release: NnuecCacheReleaseAck,
    pub observations: NnuecInitialCleanupObservations,
}

#[must_use = "poll bounded cleanup or retain it under the original deadline"]
pub struct NnuecInitialCleanup {
    phase: NnuecInitialCleanupPhase,
    custody: Option<Box<Custody>>,
}
impl NnuecInitialCleanup {
    pub fn new(cleanup: InitialNnuecCleanup) -> Self {
        let (original_phase, original_failure, owners) = cleanup.into_parts();
        Self {
            phase: NnuecInitialCleanupPhase::CancelRoot,
            custody: Some(Box::new(Custody {
                original_phase,
                original_failure,
                owners,
                observations: NnuecInitialCleanupObservations::default(),
                selected_release: None,
            })),
        }
    }
    /// No wait loop, fresh deadline, replacement owner, or destructor-based
    /// engine release. Errors advance to retained closure, preserving evidence.
    pub fn step(mut self) -> NnuecInitialCleanupStep {
        let custody = self.custody.as_deref_mut().expect("pending actual custody");
        let owners = &mut custody.owners;
        let observed = &mut custody.observations;
        match self.phase {
            NnuecInitialCleanupPhase::CancelRoot => {
                if let Some(root) = owners.root.as_mut() {
                    root.cancel();
                    observed.root_cancel_requested = true;
                }
                // A root_handoff's original cancellation is held by the
                // parent; this helper cannot fabricate access to its fields.
                self.phase = NnuecInitialCleanupPhase::StopStartingWorker;
            }
            NnuecInitialCleanupPhase::StopStartingWorker => {
                if let Some(worker) = owners.starting_worker.as_ref() {
                    worker.request_shutdown();
                    observed.starting_shutdown_requested = true;
                }
                self.phase = NnuecInitialCleanupPhase::StopReadyWorker;
            }
            NnuecInitialCleanupPhase::StopReadyWorker => {
                if let Some(worker) = owners.worker.as_ref() {
                    worker.request_shutdown();
                    observed.ready_shutdown_requested = true;
                }
                self.phase = NnuecInitialCleanupPhase::InvalidateCache;
            }
            NnuecInitialCleanupPhase::InvalidateCache => {
                match owners.actor.invalidate_context_cache() {
                    Ok(ticket) => observed.cache_invalidation = Some(ticket),
                    Err(error) => observed
                        .issues
                        .push(NnuecInitialCleanupIssue::Preparation(error)),
                }
                self.phase = NnuecInitialCleanupPhase::DrainReview;
            }
            NnuecInitialCleanupPhase::DrainReview => {
                if owners.unmatched_review.is_some() {
                    // The initial helper already collected this exact terminal;
                    // its mismatch must not trigger a second receive/discard.
                    observed.review_matches_original = Some(false);
                    owners.preparation_outstanding = false;
                } else if owners.preparation_outstanding {
                    match owners.actor.try_complete() {
                        Ok(None) => return NnuecInitialCleanupStep::Pending(self),
                        Ok(Some(completion)) => {
                            let matches = owners.expected.as_ref() == Some(&completion.provenance)
                                && match completion.result.as_ref() {
                                    Ok(bound) => {
                                        owners.expected.as_ref() == Some(bound.provenance())
                                    }
                                    Err(_) => true,
                                };
                            observed.review_matches_original = Some(matches);
                            if matches && owners.preparation_terminal_ack.is_none() {
                                let Completion {
                                    provenance,
                                    result,
                                    observation,
                                    terminal_ack,
                                } = completion;
                                observed.review_terminal_provenance = Some(provenance);
                                observed.review_terminal_observation = Some(observation);
                                observed.review_terminal_result = Some(result.map(|bound| {
                                    // No engine is materialized by this
                                    // cleanup, so release its external pin.
                                    drop(bound);
                                }));
                                owners.preparation_terminal_ack = Some(terminal_ack);
                            } else {
                                observed.drained_review = Some(completion);
                            }
                            owners.preparation_outstanding = false;
                        }
                        Err(error) => observed
                            .issues
                            .push(NnuecInitialCleanupIssue::Preparation(error)),
                    }
                }
                self.phase = NnuecInitialCleanupPhase::DrainMaterialization;
            }
            NnuecInitialCleanupPhase::DrainMaterialization => {
                if owners.unmatched_materialization.is_some() {
                    observed.materialization_matches_original = Some(false);
                    // A success still owns an unclaimed actor-side engine.
                    if owners
                        .unmatched_materialization
                        .as_ref()
                        .is_some_and(|completion| completion.result.is_err())
                    {
                        owners.materialization_outstanding = false;
                    }
                } else if owners.materialization_handoff.is_none()
                    && custody.original_phase != InitialNnuecPhase::ClaimMaterialization
                    && owners.materialization_outstanding
                {
                    match owners.actor.try_materialized() {
                        Ok(None) => return NnuecInitialCleanupStep::Pending(self),
                        Ok(Some(completion)) => {
                            let matches = owners.expected.as_ref() == Some(&completion.provenance)
                                && match completion.result.as_ref() {
                                    Ok(_) => true,
                                    Err((request, _)) => {
                                        owners.expected.as_ref() == Some(request.provenance())
                                    }
                                };
                            observed.materialization_matches_original = Some(matches);
                            if completion.result.is_err() {
                                owners.materialization_outstanding = false;
                            }
                            match completion {
                                MaterializationCompletion {
                                    provenance,
                                    result: Ok(handoff),
                                } if matches => {
                                    observed.materialization_handoff_provenance = Some(provenance);
                                    owners.materialization_handoff = Some(handoff);
                                }
                                MaterializationCompletion {
                                    provenance,
                                    result: Err((request, error)),
                                } if matches && owners.materialization_request.is_none() => {
                                    observed.materialization_handoff_provenance = Some(provenance);
                                    observed.materialization_terminal_error = Some(error);
                                    owners.materialization_request = Some(request);
                                }
                                completion => {
                                    // A mismatching terminal and any rejected
                                    // request retain their exact original custody.
                                    observed.drained_materialization = Some(completion);
                                }
                            }
                        }
                        Err(error) => observed
                            .issues
                            .push(NnuecInitialCleanupIssue::Preparation(error)),
                    }
                }
                self.phase = NnuecInitialCleanupPhase::ClaimMaterialization;
            }
            NnuecInitialCleanupPhase::ClaimMaterialization => {
                if owners.materialized_engine.is_none() {
                    if let Some(handoff) = owners.materialization_handoff.take() {
                        match handoff.claim() {
                            Ok(engine) => {
                                owners.materialized_engine = Some(engine);
                                owners.materialization_outstanding = false;
                            }
                            Err(error) => observed
                                .issues
                                .push(NnuecInitialCleanupIssue::Worker(error)),
                        }
                    }
                }
                self.phase = NnuecInitialCleanupPhase::JoinStartingWorker;
            }
            NnuecInitialCleanupPhase::JoinStartingWorker => {
                if let Some(worker) = owners.starting_worker.as_mut() {
                    match worker.try_join() {
                        Ok(None) => return NnuecInitialCleanupStep::Pending(self),
                        Ok(Some(receipt)) => observed.starting_shutdown = Some(receipt),
                        Err(error) => observed
                            .issues
                            .push(NnuecInitialCleanupIssue::Worker(error)),
                    }
                }
                self.phase = NnuecInitialCleanupPhase::RecoverStartingEngine;
            }
            NnuecInitialCleanupPhase::RecoverStartingEngine => {
                let clean_initial_join = observed
                    .starting_shutdown
                    .as_ref()
                    .is_some_and(clean_initial_shutdown);
                if clean_initial_join {
                    if let Some(worker) = owners.starting_worker.as_mut() {
                        match worker.take_joined_nnuec_retirement() {
                            Ok(engine) => observed.recovered_engine = Some(engine),
                            Err(error) => observed
                                .issues
                                .push(NnuecInitialCleanupIssue::Worker(error)),
                        }
                    }
                }
                self.phase = NnuecInitialCleanupPhase::JoinReadyWorker;
            }
            NnuecInitialCleanupPhase::JoinReadyWorker => {
                if let Some(worker) = owners.worker.as_mut() {
                    match worker.try_join() {
                        Ok(None) => return NnuecInitialCleanupStep::Pending(self),
                        Ok(Some(receipt)) => observed.ready_shutdown = Some(receipt),
                        Err(error) => observed
                            .issues
                            .push(NnuecInitialCleanupIssue::Worker(error)),
                    }
                }
                self.phase = NnuecInitialCleanupPhase::RecoverReadyEngine;
            }
            NnuecInitialCleanupPhase::RecoverReadyEngine => {
                let clean_initial_join = observed
                    .ready_shutdown
                    .as_ref()
                    .is_some_and(clean_initial_shutdown);
                if clean_initial_join && observed.recovered_engine.is_none() {
                    if let Some(worker) = owners.worker.as_mut() {
                        match worker.take_nnuec_retirement() {
                            Ok(engine) => observed.recovered_engine = Some(engine),
                            Err(error) => observed
                                .issues
                                .push(NnuecInitialCleanupIssue::Worker(error)),
                        }
                    }
                }
                self.phase = NnuecInitialCleanupPhase::SubmitMaterializationRelease;
            }
            NnuecInitialCleanupPhase::SubmitMaterializationRelease => {
                if let Some(engine) = owners.materialized_engine.take() {
                    if owners.materialization_outstanding {
                        owners.materialized_engine = Some(engine);
                        observed.issues.push(NnuecInitialCleanupIssue::Invalid(
                            "original actor materialization seat is unresolved",
                        ));
                    } else {
                        match owners.actor.try_release_materialized(engine) {
                            Ok(id) => observed.materialization_release_request = Some(id),
                            Err((PreparationError::Busy, engine)) => {
                                owners.materialized_engine = Some(engine);
                                return NnuecInitialCleanupStep::Pending(self);
                            }
                            Err((error, engine)) => {
                                owners.materialized_engine = Some(engine);
                                observed
                                    .issues
                                    .push(NnuecInitialCleanupIssue::Preparation(error));
                            }
                        }
                    }
                }
                self.phase = NnuecInitialCleanupPhase::AwaitMaterializationRelease;
            }
            NnuecInitialCleanupPhase::AwaitMaterializationRelease => {
                if let Some(expected) = observed.materialization_release_request.as_ref() {
                    match owners.actor.try_released_materialized() {
                        Ok(None) => return NnuecInitialCleanupStep::Pending(self),
                        Ok(Some(completion)) => {
                            if completion.request_id() != expected {
                                observed.unmatched_materialization_release = Some(completion);
                                observed.issues.push(NnuecInitialCleanupIssue::Invalid(
                                    "materialization release differs from original request",
                                ));
                            } else {
                                observed.materialization_release_request.take();
                                match completion.into_result() {
                                    Ok(ack) => {
                                        observed.materialization_release_matches_original = Some(
                                            owners.expected.as_ref() == Some(ack.provenance())
                                                && owners.root.as_ref().is_none_or(|root| {
                                                    ack.matches_source(root.source())
                                                }),
                                        );
                                        observed.materialization_release = Some(ack);
                                    }
                                    Err((engine, error)) => {
                                        owners.materialized_engine = Some(engine);
                                        observed
                                            .issues
                                            .push(NnuecInitialCleanupIssue::Worker(error));
                                    }
                                }
                            }
                        }
                        Err(error) => observed
                            .issues
                            .push(NnuecInitialCleanupIssue::Preparation(error)),
                    }
                }
                self.phase = NnuecInitialCleanupPhase::SubmitEngineRetirement;
            }
            NnuecInitialCleanupPhase::SubmitEngineRetirement => {
                if let Some(engine) = observed.recovered_engine.take() {
                    let matches = owners.expected.as_ref() == Some(engine.provenance())
                        && owners
                            .root
                            .as_ref()
                            .is_none_or(|root| engine.matches_source(root.source()));
                    if !matches || owners.materialization_outstanding {
                        observed.recovered_engine = Some(engine);
                        observed.issues.push(NnuecInitialCleanupIssue::Invalid(
                            "original recovered engine or actor seat is unresolved",
                        ));
                    } else {
                        match owners.actor.try_retire_engine(engine) {
                            Ok(id) => observed.retirement_request = Some(id),
                            Err((PreparationError::Busy, engine)) => {
                                observed.recovered_engine = Some(engine);
                                return NnuecInitialCleanupStep::Pending(self);
                            }
                            Err((error, engine)) => {
                                observed.recovered_engine = Some(engine);
                                observed
                                    .issues
                                    .push(NnuecInitialCleanupIssue::Preparation(error));
                            }
                        }
                    }
                }
                self.phase = NnuecInitialCleanupPhase::AwaitEngineRetirement;
            }
            NnuecInitialCleanupPhase::AwaitEngineRetirement => {
                if let Some(expected) = observed.retirement_request.as_ref() {
                    match owners.actor.try_retired_engine() {
                        Ok(None) => return NnuecInitialCleanupStep::Pending(self),
                        Ok(Some(completion)) => {
                            if completion.request_id() != expected {
                                observed.unmatched_retirement = Some(completion);
                                observed.issues.push(NnuecInitialCleanupIssue::Invalid(
                                    "engine retirement terminal differs from original request",
                                ));
                            } else {
                                observed.retirement_request.take();
                                match completion.into_result() {
                                    Ok(ack) => {
                                        observed.engine_release_matches_original = Some(
                                            owners.expected.as_ref() == Some(ack.provenance())
                                                && owners.root.as_ref().is_none_or(|root| {
                                                    ack.matches_source(root.source())
                                                }),
                                        );
                                        observed.engine_release = Some(ack);
                                    }
                                    Err((engine, error)) => {
                                        observed.recovered_engine = Some(engine);
                                        observed
                                            .issues
                                            .push(NnuecInitialCleanupIssue::Worker(error));
                                    }
                                }
                            }
                        }
                        Err(error) => observed
                            .issues
                            .push(NnuecInitialCleanupIssue::Preparation(error)),
                    }
                }
                self.phase = NnuecInitialCleanupPhase::AwaitCacheRelease;
            }
            NnuecInitialCleanupPhase::AwaitCacheRelease => {
                let actor_channel_failed = observed.issues.iter().any(|issue| {
                    matches!(
                        issue,
                        NnuecInitialCleanupIssue::Preparation(
                            PreparationError::WorkerStopped | PreparationError::WorkerPanicked
                        )
                    )
                });
                if let Some(ticket) = observed
                    .cache_invalidation
                    .as_ref()
                    .filter(|_| !actor_channel_failed)
                {
                    match owners.actor.try_context_cache_release(ticket) {
                        Ok(None) => return NnuecInitialCleanupStep::Pending(self),
                        Ok(Some(ack)) => {
                            observed.cache_clears_released_engine = observed
                                .engine_release
                                .as_ref()
                                .map(|engine| engine.matches_cache_release(&ack));
                            observed.cache_clears_released_materialization = observed
                                .materialization_release
                                .as_ref()
                                .map(|engine| engine.matches_cache_release(&ack));
                            observed.cache_release = Some(ack);
                        }
                        Err(error) => observed
                            .issues
                            .push(NnuecInitialCleanupIssue::Preparation(error)),
                    }
                }
                self.phase = NnuecInitialCleanupPhase::SelectSemanticRelease;
            }
            NnuecInitialCleanupPhase::SelectSemanticRelease => {
                let normal_stop = matches!(
                    custody.original_failure.as_ref(),
                    None | Some(InitialNnuecFailure::Preparation(
                        PreparationError::Cancelled | PreparationError::Expired
                    )) | Some(InitialNnuecFailure::Worker(
                        CpuWorkerError::NnuecAuthorityRevoked(
                            NnuecAuthorityRevocation::Cancelled | NnuecAuthorityRevocation::Expired
                        )
                    ))
                ) && observed
                    .review_terminal_result
                    .as_ref()
                    .is_none_or(|result| {
                        matches!(
                            result,
                            Ok(())
                                | Err(PreparationError::Cancelled | PreparationError::Expired)
                        )
                    })
                    // Only actual typed source cancellation/expiry can join
                    // the normal-stop path; strings and other revocations do not.
                    && observed
                        .materialization_terminal_error
                        .as_ref()
                        .is_none_or(|error| {
                            matches!(
                                error,
                                CpuWorkerError::NnuecAuthorityRevoked(
                                    NnuecAuthorityRevocation::Cancelled
                                        | NnuecAuthorityRevocation::Expired
                                )
                            )
                        })
                    && observed
                        .starting_shutdown
                        .as_ref()
                        .is_none_or(normal_initial_shutdown)
                    && observed
                        .ready_shutdown
                        .as_ref()
                        .is_none_or(normal_initial_shutdown);
                let unambiguous = normal_stop
                    && observed.issues.is_empty()
                    && observed.cache_release.is_some()
                    && observed.review_matches_original != Some(false)
                    && observed.materialization_matches_original != Some(false)
                    && observed.engine_release_matches_original != Some(false)
                    && observed.materialization_release_matches_original != Some(false)
                    && observed.cache_clears_released_engine != Some(false)
                    && observed.cache_clears_released_materialization != Some(false)
                    && observed.drained_review.is_none()
                    && observed.drained_materialization.is_none()
                    && observed.unmatched_retirement.is_none()
                    && observed.unmatched_materialization_release.is_none()
                    && observed.retirement_request.is_none()
                    && observed.materialization_release_request.is_none()
                    && observed.recovered_engine.is_none()
                    && owners.unmatched_review.is_none()
                    && owners.unmatched_materialization.is_none()
                    && !owners.preparation_outstanding
                    && !owners.materialization_outstanding
                    && owners.materialization_handoff.is_none()
                    && owners.materialized_engine.is_none()
                    && owners.root.is_some()
                    && owners.root_handoff.is_none()
                    && (owners.starting_worker.is_none()
                        || observed
                            .starting_shutdown
                            .as_ref()
                            .is_some_and(clean_initial_shutdown))
                    && (owners.worker.is_none()
                        || observed
                            .ready_shutdown
                            .as_ref()
                            .is_some_and(clean_initial_shutdown));
                if !unambiguous {
                    self.phase = NnuecInitialCleanupPhase::StopActor;
                } else if owners.original_seal.is_none()
                    && custody.original_phase == InitialNnuecPhase::PrepareReview
                    && owners.expected.is_none()
                    && owners.original_request.is_none()
                    && owners.bound.is_none()
                    && owners.preparation_terminal_ack.is_none()
                    && owners.materialization_request.is_none()
                    && owners.ready.is_none()
                    && owners.metadata.is_none()
                    && owners.checker_mirror.is_none()
                    && owners.starting_worker.is_none()
                    && owners.worker.is_none()
                {
                    // This API itself checks that the root never created its
                    // first view. A failed PrepareReview may still be too late.
                    let root = owners.root.take().expect("original root checked");
                    match root.retire_before_materialization() {
                        Ok(retired) => {
                            let cache = observed
                                .cache_release
                                .take()
                                .expect("actual cache release checked");
                            return self.ready(retired, None, cache);
                        }
                        Err((root, error)) => {
                            owners.root = Some(root);
                            observed
                                .issues
                                .push(NnuecInitialCleanupIssue::PreparingRoot(error));
                            self.phase = NnuecInitialCleanupPhase::StopActor;
                        }
                    }
                } else if owners.original_seal.is_some() {
                    custody.selected_release = if let Some(ack) = observed.engine_release.take() {
                        Some(InitialResearcherRelease::Worker(ack))
                    } else if let Some(ack) = observed.materialization_release.take() {
                        Some(InitialResearcherRelease::Materialization(ack))
                    } else if let Some(ack) = owners.preparation_terminal_ack.take() {
                        Some(InitialResearcherRelease::Preparation(ack))
                    } else {
                        owners
                            .original_request
                            .take()
                            .map(InitialResearcherRelease::Unsubmitted)
                    };
                    self.phase = if custody.selected_release.is_some() {
                        NnuecInitialCleanupPhase::ReleaseExternalPins
                    } else {
                        NnuecInitialCleanupPhase::StopActor
                    };
                } else {
                    self.phase = NnuecInitialCleanupPhase::StopActor;
                }
            }
            NnuecInitialCleanupPhase::ReleaseExternalPins => {
                // All physical engine transfers above are acknowledged. Only
                // the selected actual cleanup capability and original seal
                // cross into semantic retirement; no latest-view substitute.
                drop(owners.original_request.take());
                drop(owners.bound.take());
                drop(owners.materialization_request.take());
                drop(owners.ready.take());
                drop(owners.starting_worker.take());
                drop(owners.worker.take());
                drop(owners.metadata.take());
                drop(owners.checker_mirror.take());
                drop(owners.preparation_terminal_ack.take());
                self.phase = NnuecInitialCleanupPhase::RetireSemanticRoot;
            }
            NnuecInitialCleanupPhase::RetireSemanticRoot => {
                match (
                    owners.root.take(),
                    owners.original_seal.take(),
                    custody.selected_release.take(),
                    observed.cache_release.take(),
                ) {
                    (Some(root), Some(seal), Some(release), Some(cache)) => {
                        match root.retire_after_initial_preparation(seal, release, cache) {
                            Ok(retired) => {
                                return self.ready(
                                    retired.retired,
                                    Some(retired.release),
                                    retired.cache_release,
                                );
                            }
                            Err(rejection) => observed.semantic_rejection = Some(rejection),
                        }
                    }
                    (root, seal, release, cache) => {
                        owners.root = root;
                        owners.original_seal = seal;
                        custody.selected_release = release;
                        observed.cache_release = cache;
                        observed.issues.push(NnuecInitialCleanupIssue::Invalid(
                            "original semantic retirement custody absent",
                        ));
                    }
                }
                self.phase = NnuecInitialCleanupPhase::StopActor;
            }
            NnuecInitialCleanupPhase::StopActor => {
                owners.actor.request_shutdown();
                observed.actor_shutdown_requested = true;
                self.phase = NnuecInitialCleanupPhase::JoinActor;
            }
            NnuecInitialCleanupPhase::JoinActor => {
                match owners.actor.try_join() {
                    Ok(None) => return NnuecInitialCleanupStep::Pending(self),
                    Ok(Some(())) => {
                        observed.actor_join = Some(Ok(()));
                        // Actual actor exit releases its unclaimed engine.
                        // A retained claimed engine/Starting worker is separate.
                        owners.materialization_outstanding = false;
                    }
                    Err(error) => observed.actor_join = Some(Err(error)),
                }
                self.phase = NnuecInitialCleanupPhase::Retain;
            }
            NnuecInitialCleanupPhase::Retain => {
                return NnuecInitialCleanupStep::Retained(NnuecInitialCleanupRetained {
                    custody: self.custody.take(),
                });
            }
        }
        NnuecInitialCleanupStep::Pending(self)
    }

    fn ready(
        mut self,
        retired: RetiredPreparingRoot<AdaptivePortModel>,
        release: Option<InitialResearcherReleaseReceipt>,
        cache_release: NnuecCacheReleaseAck,
    ) -> NnuecInitialCleanupStep {
        let Custody {
            original_phase,
            original_failure,
            owners,
            observations,
            selected_release,
        } = *self.custody.take().expect("retired actual custody");
        // The success gates and typed semantic API consumed/released every
        // original semantic and physical obligation except this same actor.
        debug_assert!(selected_release.is_none());
        NnuecInitialCleanupStep::Ready(NnuecInitialCleaned {
            retired,
            actor: owners.actor,
            original_phase,
            original_failure,
            release,
            cache_release,
            observations,
        })
    }
}
impl Drop for NnuecInitialCleanup {
    fn drop(&mut self) {
        if let Some(custody) = self.custody.take() {
            std::mem::forget(custody);
        }
    }
}

fn clean_initial_shutdown(receipt: &CpuWorkerShutdown) -> bool {
    receipt.thread_joined
        && !receipt.thread_panicked
        && receipt.engine_recoverable
        && receipt.events.is_empty()
        && receipt.value_installation_receipt.is_none()
        && receipt.value_installation_error.is_none()
}

/// Share the exact initial-worker reuse gate with post-handoff cleanup.
pub(super) fn permits_initial_reuse(receipt: &CpuWorkerShutdown) -> bool {
    clean_initial_shutdown(receipt) && normal_initial_shutdown(receipt)
}

/// Engine recovery can be safe despite a rejected startup. Reuse requires
/// the separate actual Ready probe and any requested affinity restoration.
fn normal_initial_shutdown(receipt: &CpuWorkerShutdown) -> bool {
    receipt.ready_readback.as_ref().is_some_and(Result::is_ok)
        && match &receipt.thread_affinity {
            CpuWorkerThreadAffinityShutdown::NotRequested => true,
            CpuWorkerThreadAffinityShutdown::Requested { ready, finished } => {
                ready.as_ref().is_some_and(Result::is_ok)
                    && finished.as_ref().is_some_and(Result::is_ok)
            }
        }
}
