//! Explicit own-board Rovent Engine custody through the shared archive worker.
//! The session keeps its native/CPU owners while only the original Engine Box
//! moves. A logical archive result never replaces their physical root fence.

use super::*;
use crate::rovent_bootstrap::launch::storage::RoventStorageOwnedRuntimeRegistration;
use crate::rovent_bootstrap::product::SharedRoventStorageRegistration;
use crate::rovent_bootstrap::storage_lifecycle::{SessionStorageLifecycle, StorageSemanticOwner};

pub(super) struct EngineStorageSlot {
    lifecycle: Box<SessionStorageLifecycle<AdaptivePortModel>>,
    // Read from the actual returned Engine, never reconstructed from a go.
    last_authority: Option<SearchAuthority>,
    caller_panic: Option<Box<dyn std::any::Any + Send>>,
}

pub(super) fn original_storage_admission(
    admission: Option<&RoventPcrRootAdmission>,
) -> Result<Arc<crate::rovent_root_admission::RoventRootAdmission>, SearchSessionFailure> {
    admission
        .and_then(|admission| admission.original.as_ref())
        .cloned()
        .ok_or_else(|| {
            failure(
                "RoventEngineStorageAdmission",
                "original opaque per-go admission is required",
            )
        })
}

pub(super) fn unused_engine_storage_owner<B>(owners: &IdleOwners<B>) -> bool
where
    B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
{
    matches!(
        owners.rovent_idle_custody,
        RoventPcrIdleCustody::StartupReady
    ) && owners.engine.as_ref().is_some_and(|engine| {
        engine.last_adaptive_authority().is_none()
            && engine.stores().generation() == 0
            && engine.stores().root().is_none()
            && engine.stores().hot_stats() == rz_search::pals::store::ArchiveStats::default()
            && !engine.archive_enabled()
    }) && owners.unexpected_archive_owner.is_none()
        && owners.continuation.is_none()
        && owners.native_finished.is_none()
        && !owners.native_joined
        && !owners.native.is_quarantined()
        && owners.native.occupancy() == 0
        && !owners.cpu.is_empty()
        && owners.external.is_none()
        && matches!(owners.researcher_source, SessionResearcherSource::Legacy)
        && owners.profiled_retirement.is_none()
}

impl<B> AdaptiveSessionDriver<B>
where
    B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
    B::Output: ProposerRankingOutput,
{
    pub(crate) fn storage_observation(
        &self,
    ) -> Result<
        Option<crate::rovent_bootstrap::storage_lifecycle::StorageLifecycleObservation>,
        &'static str,
    > {
        let slot = self
            .storage
            .try_lock()
            .map_err(|_| "storage_lifecycle_busy_or_poisoned")?;
        Ok(slot.as_ref().map(|slot| {
            let mut observation = slot.lifecycle.observation();
            if slot.caller_panic.is_some() {
                observation.complete = false;
            }
            observation
        }))
    }
    #[allow(clippy::result_large_err)]
    pub(crate) fn install_storage_registration(
        &self,
        original: RoventStorageOwnedRuntimeRegistration,
    ) -> Result<
        Arc<SharedRoventStorageRegistration>,
        (RoventStorageOwnedRuntimeRegistration, SearchSessionFailure),
    > {
        let checked = (|| {
            let state = self.state.try_lock().map_err(|_| {
                failure(
                    "RoventEngineStorageInstall",
                    "original session owner is busy/poisoned",
                )
                .unknown_if_unclosed()
            })?;
            let slot = self.storage.try_lock().map_err(|_| {
                failure(
                    "RoventEngineStorageInstall",
                    "original storage slot is busy/poisoned",
                )
                .unknown_if_unclosed()
            })?;
            let policy = self.rovent_root_policy.as_ref().ok_or_else(|| {
                failure(
                    "RoventEngineStorageInstall",
                    "original Rovent root policy is absent",
                )
            })?;
            let no_prior_go = self
                .latest_root
                .try_lock()
                .map_err(|_| {
                    failure(
                        "RoventEngineStorageInstall",
                        "root receipt is busy/poisoned",
                    )
                })?
                .is_none()
                && self
                    .latest_rovent_root
                    .try_lock()
                    .map_err(|_| {
                        failure(
                            "RoventEngineStorageInstall",
                            "Rovent root receipt is busy/poisoned",
                        )
                    })?
                    .is_none()
                && self
                    .latest_rovent_initial
                    .try_lock()
                    .map_err(|_| {
                        failure(
                            "RoventEngineStorageInstall",
                            "initial source receipt is busy/poisoned",
                        )
                    })?
                    .is_none()
                && self
                    .pending_output
                    .try_lock()
                    .map_err(|_| {
                        failure(
                            "RoventEngineStorageInstall",
                            "original UCI output is busy/poisoned",
                        )
                    })?
                    .is_none();
            let own_board = self.rovent.as_ref().is_some_and(|profile| {
                profile.mode.has_native_r()
                    && matches!(
                        policy.lock.manifest.engine.researcher,
                        Some(rz_experiments::RoventResearcher::Own { .. })
                    )
                    && policy.lock.manifest.engine.nnuec.is_none()
            });
            let valid = matches!(&*state, SessionState::Idle(owners)
                if unused_engine_storage_owner(owners)
                    && owners.native.ready_clock().is_some_and(|clock|
                        clock.domain() == self.clock.domain()
                            && clock.origin() == self.clock.runtime_clock().origin()));
            if !own_board
                || !matches!(self.researcher_class, SessionResearcherClass::Legacy)
                || !matches!(self.registration.identity, CheckerIdentity::Owned(_))
                || slot.is_some()
                || self.closed.load(Ordering::Acquire)
                || self.reset_requests.load(Ordering::Acquire) != 0
                || !valid
                || !no_prior_go
            {
                return Err(failure(
                    "RoventEngineStorageInstall",
                    "storage requires the unused own-board native Rovent Engine owner",
                ));
            }
            original
                .validate_session_attachment(&policy.lock, &self.clock, self.options.implementation)
                .map_err(|error| {
                    SearchSessionFailure::debug("RoventEngineStorageNamespace", &error)
                })?;
            original
                .profile()
                .validate_for_manifest(&policy.lock.manifest)
                .map_err(|error| {
                    SearchSessionFailure::debug("RoventEngineStorageResources", &error)
                })?;
            Ok((state, slot))
        })();
        let (_state, mut slot) = match checked {
            Ok(guards) => guards,
            Err(error) => return Err((original, error)),
        };
        // All rejection paths precede this sole move and same-Arc publication.
        let shared = Arc::new(SharedRoventStorageRegistration::new(original));
        *slot = Some(EngineStorageSlot {
            lifecycle: SessionStorageLifecycle::new(Arc::clone(&shared)),
            last_authority: None,
            caller_panic: None,
        });
        Ok(shared)
    }

    /// The source has passed into_initial before this method is reached. A
    /// stopped-initial source never resets a CPU or takes the Engine Box.
    pub(super) fn prepare_engine_storage(
        &self,
        owners: &mut IdleOwners<B>,
        admission: Option<&RoventPcrRootAdmission>,
        reset: bool,
        position: &Position,
        context: &SearchSessionContext,
        initial_game_bound: &mut bool,
    ) -> Result<bool, SearchSessionFailure> {
        let mut slot = self.storage.try_lock().map_err(|_| {
            failure(
                "RoventEngineStorage",
                "original storage slot is busy/poisoned",
            )
            .unknown_if_unclosed()
        })?;
        let Some(storage) = slot.as_mut() else {
            return Ok(false);
        };
        let original = original_storage_admission(admission)?;
        let initial_setup = !storage.lifecycle.configured();
        if initial_setup {
            // Keep the same borrowed IdleOwners even if a model preflight or
            // actual reset panics before generic worker custody can adopt it.
            match catch_unwind(AssertUnwindSafe(|| {
                self.prepare_initial_engine_archive_game(owners, context, reset, initial_game_bound)
            })) {
                Ok(result) => result?,
                Err(payload) => {
                    storage.caller_panic.get_or_insert(payload);
                    return Err(failure(
                        "RoventEngineStorageInitialPanic",
                        "original initial-game panic and owners retained",
                    )
                    .unknown_if_unclosed());
                }
            }
        } else if reset {
            // Configured NewGame consumes this actual arm in the worker hook.
            self.reset_actual_cpu(owners, context, initial_game_bound)?;
        }
        let engine = owners
            .engine
            .take()
            .ok_or_else(SearchSessionFailure::physical_completion_unknown)?;
        storage.last_authority = engine.last_adaptive_authority();
        let signal = owners.native.signal();
        match storage.lifecycle.prepare(
            StorageSemanticOwner::EnginePcr(engine),
            original,
            reset,
            position,
            &signal,
        ) {
            Ok(StorageSemanticOwner::EnginePcr(engine)) => {
                owners.engine = Some(engine);
                Ok(true)
            }
            Ok(other) => {
                storage.lifecycle.retain_wrong_return(other);
                Err(
                    failure("RoventEngineStorageReturn", "original Engine kind differs")
                        .unknown_if_unclosed(),
                )
            }
            Err(mut rejected) => {
                if let Some(owner) = rejected.owner.take() {
                    match owner {
                        StorageSemanticOwner::EnginePcr(engine) => owners.engine = Some(engine),
                        other => storage.lifecycle.retain_wrong_return(other),
                    }
                }
                Err(rejected.failure.unknown_if_unclosed())
            }
        }
    }

    pub(super) fn prepare_initial_engine_archive_game(
        &self,
        owners: &mut IdleOwners<B>,
        context: &SearchSessionContext,
        game_changed: bool,
        initial_game_bound: &mut bool,
    ) -> Result<(), SearchSessionFailure> {
        self.reset_actual_cpu(owners, context, initial_game_bound)?;
        let engine = owners
            .engine
            .as_mut()
            .ok_or_else(SearchSessionFailure::physical_completion_unknown)?;
        if game_changed {
            // Preserve the original strict next-game path and its one mirror
            // reset. The exact-ready check below then performs no reset.
            engine
                .try_adaptive_new_game(
                    context.authority,
                    context.control.deadline,
                    &context.control.cancellation,
                )
                .map_err(|error| {
                    SearchSessionFailure::debug("RoventEngineStorageInitialGame", &error)
                })?;
        }
        // At equal initial game0 the unused-only entry consumes the original
        // mirror arm once. An existing exact ready is checked without reset.
        engine
            .prepare_unused_archive_game(
                context.authority,
                context.control.deadline,
                &context.control.cancellation,
            )
            .map_err(|error| SearchSessionFailure::debug("RoventEngineStorageInitialGame", &error))
    }

    /// Native/CPU physical closure is a caller precondition. A successful final
    /// Idle stays in this lifecycle after its real checkpoint/close/join; it
    /// cannot be restored as a searchable Engine. Some(None) means selected
    /// storage before any registered root, not fabricated root authority.
    pub(super) fn finish_engine_storage(
        &self,
        owners: &mut IdleOwners<B>,
        until: Instant,
    ) -> Result<Option<Option<SearchAuthority>>, SearchSessionFailure> {
        let mut slot = self.storage.try_lock().map_err(|_| {
            failure(
                "RoventEngineStorageFinish",
                "original storage slot is busy/poisoned",
            )
            .unknown_if_unclosed()
        })?;
        let Some(storage) = slot.as_mut() else {
            return Ok(None);
        };
        let original = owners.engine.take().map(|engine| {
            storage.last_authority = engine.last_adaptive_authority();
            StorageSemanticOwner::EnginePcr(engine)
        });
        match storage
            .lifecycle
            .finish(original, until, &owners.native.signal())
        {
            Ok(()) => Ok(Some(storage.last_authority)),
            Err(mut rejected) => {
                if let Some(owner) = rejected.owner.take() {
                    match owner {
                        StorageSemanticOwner::EnginePcr(engine) => owners.engine = Some(engine),
                        other => storage.lifecycle.retain_wrong_return(other),
                    }
                }
                Err(rejected.failure.unknown_if_unclosed())
            }
        }
    }

    pub(super) fn abort_engine_storage(
        &self,
        original: SearchSessionFailure,
        owners: &IdleOwners<B>,
        until: Instant,
    ) -> SearchSessionFailure {
        match self.storage.try_lock() {
            Ok(mut slot) => match slot.as_mut() {
                Some(storage) => storage
                    .lifecycle
                    .abort_and_join(
                        original.clone().unknown_if_unclosed(),
                        until,
                        &owners.native.signal(),
                    )
                    .err()
                    .unwrap_or(original)
                    .unknown_if_unclosed(),
                None => original,
            },
            Err(_) => original.unknown_if_unclosed(),
        }
    }
}
