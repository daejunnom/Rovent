//! Explicit own-board Rovent Engine custody through the shared archive worker.
//! The session keeps its native/CPU owners while only the original Engine Box
//! moves. A logical archive result never replaces their physical root fence.

use super::*;
use crate::rovent_bootstrap::launch::storage::{
    RoventStorageOwnedRuntimeRegistration, RoventStorageOwnedRuntimeRegistrationV2,
    RoventStoragePreparedTransferV3,
};
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

enum OriginalStorageAdmission {
    V1(Arc<crate::rovent_root_admission::RoventRootAdmission>),
    V2(Arc<crate::rovent_root_admission::source_seal::RoventRootAdmissionV2>),
    V3(Arc<crate::rovent_root_admission::source_seal::RoventRootAdmissionV3>),
}

fn original_storage_admission_versioned(
    admission: Option<&RoventPcrRootAdmission>,
) -> Result<OriginalStorageAdmission, SearchSessionFailure> {
    if let Some(original) = admission.and_then(|value| value.original_v3.as_ref()) {
        return Ok(OriginalStorageAdmission::V3(Arc::clone(original)));
    }
    if let Some(original) = admission.and_then(|value| value.original_v2.as_ref()) {
        return Ok(OriginalStorageAdmission::V2(Arc::clone(original)));
    }
    original_storage_admission(admission).map(OriginalStorageAdmission::V1)
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
                profile.common().mode.has_native_r()
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

    /// Install the same move-only registration for the original /2 session.
    /// The /2 root policy and process clock are checked before the owner moves.
    #[allow(clippy::result_large_err)]
    pub(crate) fn install_storage_registration_v2(
        &self,
        original: RoventStorageOwnedRuntimeRegistrationV2,
    ) -> Result<
        Arc<SharedRoventStorageRegistration>,
        (
            RoventStorageOwnedRuntimeRegistrationV2,
            SearchSessionFailure,
        ),
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
            let policy = self.rovent_root_policy_v2.as_ref().ok_or_else(|| {
                failure(
                    "RoventEngineStorageInstall",
                    "original /2 root policy is absent",
                )
            })?;
            let lock = policy.owner.v2().ok_or_else(|| {
                failure("RoventEngineStorageInstall", "original /2 lock is absent")
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
                    .latest_rovent_root_v2
                    .try_lock()
                    .map_err(|_| {
                        failure(
                            "RoventEngineStorageInstall",
                            "/2 root receipt is busy/poisoned",
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
                    .latest_rovent_initial_v2
                    .try_lock()
                    .map_err(|_| {
                        failure(
                            "RoventEngineStorageInstall",
                            "/2 initial source receipt is busy/poisoned",
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
                profile.common().mode.has_native_r()
                    && matches!(
                        lock.manifest.engine.researcher,
                        Some(rz_experiments::RoventResearcher::Own { .. })
                    )
                    && lock.manifest.engine.nnuec.is_none()
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
                    "/2 storage requires the unused own-board native Rovent Engine owner",
                ));
            }
            original
                .validate_session_attachment(lock, &self.clock, self.options.implementation)
                .map_err(|error| {
                    SearchSessionFailure::debug("RoventEngineStorageNamespace", &error)
                })?;
            original
                .profile()
                .validate_for_manifest_v2(&lock.manifest)
                .map_err(|error| {
                    SearchSessionFailure::debug("RoventEngineStorageResources", &error)
                })?;
            Ok((state, slot))
        })();
        let (_state, mut slot) = match checked {
            Ok(guards) => guards,
            Err(error) => return Err((original, error)),
        };
        let shared = Arc::new(SharedRoventStorageRegistration::new_v2(original));
        *slot = Some(EngineStorageSlot {
            lifecycle: SessionStorageLifecycle::new(Arc::clone(&shared)),
            last_authority: None,
            caller_panic: None,
        });
        Ok(shared)
    }

    /// Install into the exact idle V3 session while the original transfer slot
    /// remains pinned by its launcher scope. All allocation and validation
    /// precede the move-only transfer commit.
    pub(crate) fn install_prepared_storage_registration_v3(
        &self,
        shared: Arc<SharedRoventStorageRegistration>,
        prepared: RoventStoragePreparedTransferV3,
    ) -> Result<(), SearchSessionFailure> {
        let unknown = |detail: &'static str| {
            failure("RoventEngineStorageInstallV3", detail).unknown_if_unclosed()
        };
        let state = self
            .state
            .try_lock()
            .map_err(|_| unknown("session busy or poisoned"))?;
        let mut slot = self
            .storage
            .try_lock()
            .map_err(|_| unknown("storage busy or poisoned"))?;
        let policy = self
            .rovent_root_policy_v3
            .as_ref()
            .ok_or_else(|| unknown("V3 root policy absent"))?;
        let lock = policy
            .owner
            .v3()
            .ok_or_else(|| unknown("original V3 lock absent"))?;
        let valid_idle = matches!(&*state, SessionState::Idle(owners)
            if unused_engine_storage_owner(owners)
                && owners.native.ready_clock().is_some_and(|clock|
                    clock.domain() == self.clock.domain()
                        && clock.origin() == self.clock.runtime_clock().origin()));
        let clock = self.clock.runtime_clock();
        let expected = prepared.clock().runtime_clock();
        let own_board = self.rovent.as_ref().is_some_and(|profile| {
            profile.common().mode.has_native_r()
                && matches!(
                    lock.manifest.engine.researcher,
                    Some(rz_experiments::RoventResearcher::Own { .. })
                )
                && lock.manifest.engine.nnuec.is_none()
        });
        let no_prior_go = self
            .latest_root
            .try_lock()
            .map_err(|_| unknown("root busy"))?
            .is_none()
            && self
                .latest_rovent_root
                .try_lock()
                .map_err(|_| unknown("root busy"))?
                .is_none()
            && self
                .latest_rovent_root_v2
                .try_lock()
                .map_err(|_| unknown("V2 root busy"))?
                .is_none()
            && self
                .latest_rovent_root_v3
                .try_lock()
                .map_err(|_| unknown("V3 root busy"))?
                .is_none()
            && self
                .latest_rovent_initial
                .try_lock()
                .map_err(|_| unknown("initial busy"))?
                .is_none()
            && self
                .latest_rovent_initial_v2
                .try_lock()
                .map_err(|_| unknown("V2 initial busy"))?
                .is_none()
            && self
                .latest_rovent_initial_v3
                .try_lock()
                .map_err(|_| unknown("V3 initial busy"))?
                .is_none()
            && self
                .pending_output
                .try_lock()
                .map_err(|_| unknown("output busy"))?
                .is_none();
        if !own_board
            || !matches!(self.researcher_class, SessionResearcherClass::Legacy)
            || !matches!(self.registration.identity, CheckerIdentity::Owned(_))
            || slot.is_some()
            || self.closed.load(Ordering::Acquire)
            || self.reset_requests.load(Ordering::Acquire) != 0
            || !valid_idle
            || !no_prior_go
            || !Arc::ptr_eq(lock, prepared.input_lock())
            || self.clock.epoch() != prepared.clock().epoch()
            || clock.domain() != expected.domain()
            || clock.origin() != expected.origin()
            || self.options.implementation != prepared.implementation()
        {
            return Err(unknown(
                "V3 storage requires the same unused original Own session",
            ));
        }
        prepared
            .profile()
            .validate_for_manifest_v3(&lock.manifest)
            .map_err(|error| {
                SearchSessionFailure::debug("RoventEngineStorageResourcesV3", &error)
                    .unknown_if_unclosed()
            })?;
        let lifecycle = SessionStorageLifecycle::new(shared);
        prepared.commit().map_err(|error| {
            SearchSessionFailure::debug("RoventEngineStorageTransferV3", &error)
                .unknown_if_unclosed()
        })?;
        *slot = Some(EngineStorageSlot {
            lifecycle,
            last_authority: None,
            caller_panic: None,
        });
        Ok(())
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
        let original = original_storage_admission_versioned(admission)?;
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
        let owner = StorageSemanticOwner::EnginePcr(engine);
        let prepared = match original {
            OriginalStorageAdmission::V1(original) => storage
                .lifecycle
                .prepare(owner, original, reset, position, &signal),
            OriginalStorageAdmission::V2(original) => storage
                .lifecycle
                .prepare_v2(owner, original, reset, position, &signal),
            OriginalStorageAdmission::V3(original) => storage
                .lifecycle
                .prepare_v3(owner, original, reset, position, &signal),
        };
        match prepared {
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
