//! Typed external-R admission over the shared session and Store lifecycle.
//!
//! The dormant source retains the exact executable/cwd descriptors. No process
//! exists until the first original, live UCI root has been admitted. CPU_T and
//! the legacy external-worker entry points cannot construct this source kind.

use super::*;
use crate::pals_adaptive::{
    ExternalManagerLock, ExternalManagerRoot, ProfiledExternalManagerSource,
};
use crate::rovent_bootstrap::{
    ExternalNativeInputLock, RoventExternalRSourceCustody, RoventPinnedExternalRSource,
};
use crate::rovent_external_launch::{
    PreparedRoventExternalLaunch, RoventAdmittedExternalRootSource,
    RoventAdmittedExternalRootSourceV2, RoventExternalOwner, RoventExternalSpawnRejection,
};
use crate::rovent_root_admission::RoventRootAdmission;
use rz_contracts::pals_adaptive::CpuWorkerId;
use rz_experiments::{RoventExternalPurpose, RoventInputLock, RoventInputLockV2};
use rz_runtime::contracts::ContractClock as _;
use rz_search::cpu_checker::CheckerError;
use rz_search::external_cpu::ProfiledExternalRootSource;
use rz_search::external_cpu::worker::ExternalWorkerClock;
use rz_search::pals::engine::adaptive_port::AdaptiveFrontierStartup;
use rz_search::pals::engine::adaptive_port::profiled_retirement_archive::RegisteredProfiledRetirementSink;
use rz_search::pals::rovent_engine::RootMoveDomain;
use rz_search::pals::rovent_frontier::FrontierRegistration;
use std::fs::File;

/// Separate typed constructor input; it is never accepted by the old public
/// own-R/legacy `AdaptiveSessionSources` constructor.
pub struct RoventExternalSessionSources<B>
where
    B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
{
    pub common: AdaptiveSessionSources<B>,
    pub external: RoventExternalSessionSource,
}

pub struct RoventExternalSessionSourcesV2<B>
where
    B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
{
    pub common: AdaptiveSessionSources<B>,
    pub external: RoventExternalSessionSource<RoventInputLockV2>,
}

pub struct RoventExternalSessionAdmissionRejectionV2<B>
where
    B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
{
    pub failure: SearchSessionFailure,
    pub sources: AdaptiveSessionSources<B>,
    pub external: RoventExternalSessionSource<RoventInputLockV2>,
    pub retirement_sink: RegisteredProfiledRetirementSink,
}

pub struct RoventExternalSessionAdmissionRejection<B>
where
    B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
{
    pub failure: SearchSessionFailure,
    pub sources: AdaptiveSessionSources<B>,
    pub external: RoventExternalSessionSource,
    pub retirement_sink: RegisteredProfiledRetirementSink,
}

struct DormantExternalSource {
    launch: PreparedRoventExternalLaunch,
    executable: File,
    working_directory: File,
}

pub struct RoventExternalSourcePreparationRejection<L: SessionExternalLock = RoventInputLock> {
    pub error: CheckerError,
    _retained: RejectedExternalSource<L>,
}

enum RejectedExternalSource<L: SessionExternalLock> {
    Pinned {
        _source: RoventPinnedExternalRSource<L>,
    },
    Consumed {
        _launch: PreparedRoventExternalLaunch,
        _executable: File,
        _working_directory: File,
        _custody: RoventExternalRSourceCustody<L>,
    },
}

// Each session source has one process-state slot. Original launch/FD/worker
// custody moves between these variants rather than entering an unbounded queue.
#[allow(clippy::large_enum_variant)]
enum ExternalProcessState<L: SessionExternalLock> {
    Dormant(DormantExternalSource),
    Running(RoventExternalOwner),
    /// The retaining startup entry guarantees that no physical worker was
    /// created on this path. Preserve its actual error and original FDs rather
    /// than retrying by reopening paths or calling this an unknown live worker.
    SpawnRejected(RoventExternalSpawnRejection),
    /// Exclusive ownership is being transferred. A panic cannot turn this
    /// unknown boundary into permission to reopen paths or start a new worker.
    Failed,
    /// Returned by the common manager. The helper still owns its original
    /// process/custody until its actual fence permits extracting them.
    Returned(Box<ProfiledExternalManagerSource<L>>),
}

/// Unique retained source owner. The public type has no public constructor or
/// fields; only the pinned, purpose-checked input above can create it.
pub struct RoventExternalSessionSource<L: SessionExternalLock = RoventInputLock> {
    state: ExternalProcessState<L>,
    custody: Option<RoventExternalRSourceCustody<L>>,
    declaration: AdaptiveCheckerMetadata,
    input_lock: Digest,
    canonical_profile: String,
    capability: String,
    admission: Option<rz_search::external_cpu::ProfiledResearcherAdmission>,
    shutdown: Option<CheckerShutdown>,
}

pub trait SessionExternalLock:
    Sized + ExternalNativeInputLock + ExternalManagerLock<Custody = RoventExternalRSourceCustody<Self>>
{
}
impl SessionExternalLock for RoventInputLock {}
impl SessionExternalLock for RoventInputLockV2 {}

impl<L: SessionExternalLock> RoventExternalSessionSource<L> {
    // A rejected preparation returns its original source/FD custody without
    // allocating a replacement error owner on the failure path.
    #[allow(clippy::result_large_err)]
    pub(crate) fn from_pinned(
        external: RoventPinnedExternalRSource<L>,
    ) -> Result<Self, RoventExternalSourcePreparationRejection<L>> {
        if external.source_purpose() != RoventExternalPurpose::ResearchChecker {
            return Err(RoventExternalSourcePreparationRejection {
                error: CheckerError::Invalid(
                    "external session requires the original researcher source",
                ),
                _retained: RejectedExternalSource::Pinned { _source: external },
            });
        }
        // Source admission has already verified the profile and the held FDs.
        // Read-only declaration access below performs no process startup.
        let input_lock = external.input_lock_digest();
        let (launch, _profile_bytes, executable, working_directory, custody) =
            external.into_parts();
        let declarations = launch.checker_declaration().and_then(|checker| {
            let admission = checker.profiled_researcher_admission()?;
            let declaration = AdaptiveCheckerMetadata::capture(&checker)?;
            Ok((declaration, admission))
        });
        let (declaration, admission) = match declarations {
            Ok(declarations) => declarations,
            Err(error) => {
                // The consumed source cannot be reconstituted from paths.
                // This is handled by the retaining factory before this method.
                return Err(RoventExternalSourcePreparationRejection {
                    error,
                    _retained: RejectedExternalSource::Consumed {
                        _launch: launch,
                        _executable: executable,
                        _working_directory: working_directory,
                        _custody: custody,
                    },
                });
            }
        };
        Ok(Self {
            input_lock,
            canonical_profile: launch.canonical_sha256().into(),
            capability: launch.capability_sha256().into(),
            state: ExternalProcessState::Dormant(DormantExternalSource {
                launch,
                executable,
                working_directory,
            }),
            custody: Some(custody),
            declaration,
            admission: Some(admission),
            shutdown: None,
        })
    }

    pub(crate) fn declaration(&self) -> &AdaptiveCheckerMetadata {
        &self.declaration
    }
    pub(crate) fn admission(&self) -> &rz_search::external_cpu::ProfiledResearcherAdmission {
        self.admission
            .as_ref()
            .expect("original dormant declaration")
    }

    pub(super) fn validate_declaration<B>(
        &self,
        profile: &RoventPcrSessionProfile,
        sources: &AdaptiveSessionSources<B>,
    ) -> Result<(), SearchSessionFailure>
    where
        B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
    {
        if !profile.mode.has_native_r()
            || profile.input_lock != self.input_lock
            || profile.maximum_goal_plies != profile.frontier_limits.maximum_goal_plies
            || profile.reciprocal.mode != profile.mode
            || profile.reciprocal.root_budget != profile.root_budget
            || !sources.cpu.is_empty()
            || sources.external.is_some()
            || sources.engine.checker_identity() != self.declaration.identity()
            || sources.checker_metadata.identity() != self.declaration.identity()
            || sources.checker_metadata.conditions() != self.declaration.conditions()
            || sources.checker_metadata.capabilities() != self.declaration.capabilities()
            || !matches!(self.declaration.identity(), CheckerIdentity::ExternalUci(_))
            || sources.engine.resolver_policy()
                != rz_search::pals::engine::ResolverPolicy::ModelWdlRestricted
        {
            return Err(failure(
                "RoventExternalAssembly",
                "external R requires its separate pinned source and no native R workers",
            ));
        }
        profile
            .model_namespace
            .verify_identity(
                sources.engine.model_identity(),
                sources.engine.model_value_identity(),
            )
            .map_err(|error| SearchSessionFailure::debug("RoventExternalModelNamespace", &error))
    }

    pub(super) fn prepare_actual_root(
        &mut self,
        root: Arc<L::Root>,
        clock: &ProcessClock,
    ) -> Result<(), SearchSessionFailure> {
        if self
            .custody
            .as_ref()
            .is_some_and(|custody| !L::matches_owner(custody, &root))
            || root.startup_lock_digest() != self.input_lock
            || root.runtime_clock().domain() != clock.domain()
            || root.runtime_clock().origin() != clock.runtime_clock().origin()
            || !root.original_context().accepts().unwrap_or(false)
        {
            return Err(failure(
                "RoventExternalRoot",
                "foreign or revoked original root",
            ));
        }
        self.extract_returned_owner()?;
        if matches!(self.state, ExternalProcessState::Dormant(_)) {
            let ExternalProcessState::Dormant(pending) =
                std::mem::replace(&mut self.state, ExternalProcessState::Failed)
            else {
                unreachable!()
            };
            let worker_clock = ExternalWorkerClock {
                domain: clock.domain(),
                origin: clock.runtime_clock().origin(),
            };
            let started = pending.launch.spawn_pinned_retaining(
                pending.executable,
                pending.working_directory,
                CpuWorkerId(1),
                worker_clock,
                root.authority(),
                root.original_context().control.deadline,
                &root.original_context().control.cancellation,
            );
            let owner = match started {
                Ok(owner) => owner,
                Err(rejected) => {
                    let error =
                        SearchSessionFailure::debug("RoventExternalSpawn", &rejected.failure);
                    self.state = ExternalProcessState::SpawnRejected(rejected);
                    return Err(error);
                }
            };
            self.state = ExternalProcessState::Running(owner);
        }
        if let ExternalProcessState::SpawnRejected(rejected) = &self.state {
            return Err(SearchSessionFailure::debug(
                "RoventExternalSpawn",
                &rejected.failure,
            ));
        }
        let ExternalProcessState::Running(owner) = &mut self.state else {
            return Err(SearchSessionFailure::physical_completion_unknown());
        };
        loop {
            if let Some(startup) = owner
                .verified_startup()
                .map_err(|error| SearchSessionFailure::debug("RoventExternalStartup", &error))?
            {
                let input_lock: String = self
                    .input_lock
                    .0
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect();
                if startup.input_lock_sha256() != input_lock
                    || startup.canonical_sha256() != self.canonical_profile
                    || startup.capability_sha256() != self.capability
                {
                    return Err(failure(
                        "RoventExternalStartup",
                        "actual source changed at startup",
                    ));
                }
                break;
            }
            if !root.original_context().accepts().unwrap_or(false)
                || Instant::now() >= root.original_context().control.deadline
            {
                owner
                    .worker()
                    .request_shutdown(root.original_context().shutdown_deadline);
                return Err(failure(
                    "RoventExternalStartup",
                    "startup canceled or expired",
                ));
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        if (owner.actual_root_admission().is_some() || owner.actual_root_admission_v2().is_some())
            && !L::is_current(owner, &root)
        {
            owner
                .worker()
                .request_root_change(root.authority(), root.original_context().control.deadline)
                .map_err(|error| {
                    SearchSessionFailure::debug("RoventExternalRootChange", &error)
                })?;
            loop {
                if let Some(result) = owner.worker().root_change_result() {
                    result.map_err(|error| {
                        SearchSessionFailure::debug("RoventExternalRootChange", &error)
                    })?;
                    break;
                }
                if !root.original_context().accepts().unwrap_or(false)
                    || Instant::now() >= root.original_context().control.deadline
                {
                    return Err(failure(
                        "RoventExternalRootChange",
                        "root change canceled or expired",
                    ));
                }
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        L::admit(owner, root)
            .map_err(|error| SearchSessionFailure::debug("RoventExternalRoot", &error))
    }

    fn extract_returned_owner(&mut self) -> Result<(), SearchSessionFailure> {
        if !matches!(self.state, ExternalProcessState::Returned(_)) {
            return Ok(());
        }
        let ExternalProcessState::Returned(source) =
            std::mem::replace(&mut self.state, ExternalProcessState::Failed)
        else {
            unreachable!()
        };
        match source.into_parts() {
            Ok((owner, custody)) => {
                self.custody = Some(custody);
                self.state = ExternalProcessState::Running(owner);
                Ok(())
            }
            Err(source) => {
                self.state = ExternalProcessState::Returned(Box::new(source));
                Err(SearchSessionFailure::physical_completion_unknown())
            }
        }
    }

    // Failed adoption returns this same session source and actual error,
    // preserving launch/worker/FD ownership without a fresh failure allocation.
    #[allow(clippy::result_large_err)]
    pub(super) fn into_manager(
        mut self,
        root: Arc<L::Root>,
    ) -> Result<ProfiledExternalManagerSource<L>, (Self, SearchSessionFailure)> {
        if let Err(error) = self.extract_returned_owner() {
            return Err((self, error));
        }
        let previous = std::mem::replace(&mut self.state, ExternalProcessState::Failed);
        let ExternalProcessState::Running(owner) = previous else {
            self.state = previous;
            return Err((self, SearchSessionFailure::physical_completion_unknown()));
        };
        let Some(custody) = self.custody.take() else {
            self.state = ExternalProcessState::Running(owner);
            return Err((self, SearchSessionFailure::physical_completion_unknown()));
        };
        match ProfiledExternalManagerSource::<L>::from_ready(owner, custody, root) {
            Ok(source) => Ok(source),
            Err((owner, custody, error)) => {
                self.state = ExternalProcessState::Running(owner);
                self.custody = Some(custody);
                Err((
                    self,
                    SearchSessionFailure::debug("RoventExternalManagerSource", &error),
                ))
            }
        }
    }

    pub(super) fn from_manager(
        source: ProfiledExternalManagerSource<L>,
        declaration: AdaptiveCheckerMetadata,
    ) -> Self {
        let shutdown = source
            .shutdown_snapshot()
            .and_then(|receipt| receipt.as_ref().ok())
            .copied();
        Self {
            input_lock: source.root_admission().startup_lock_digest(),
            canonical_profile: source.startup().canonical_sha256().into(),
            capability: source.startup().capability_sha256().into(),
            state: ExternalProcessState::Returned(Box::new(source)),
            custody: None,
            declaration,
            admission: None,
            shutdown,
        }
    }

    pub(super) fn running_owner(&self) -> Option<&RoventExternalOwner> {
        match &self.state {
            ExternalProcessState::Running(owner) => Some(owner),
            ExternalProcessState::Returned(source) => Some(source.owner()),
            _ => None,
        }
    }

    /// Historical source identity remains available after actual root/process
    /// closure. It grants no live admission or root reuse on its own.
    pub(super) fn retained_root_source(&self) -> Option<&ProfiledExternalRootSource> {
        match &self.state {
            ExternalProcessState::Returned(source) => Some(source.current_source()),
            _ => None,
        }
    }

    pub(super) fn begin_process_finish(&mut self, until: Instant) {
        match &mut self.state {
            ExternalProcessState::Running(owner) => owner.worker().request_shutdown(until),
            ExternalProcessState::Returned(source) => source.request_shutdown(until),
            ExternalProcessState::Dormant(_)
            | ExternalProcessState::SpawnRejected(_)
            | ExternalProcessState::Failed => {}
        }
    }

    pub(super) fn try_process_finish(&mut self) -> Result<bool, SearchSessionFailure> {
        let joined = match &mut self.state {
            ExternalProcessState::Dormant(_) | ExternalProcessState::SpawnRejected(_) => {
                return Ok(true);
            }
            ExternalProcessState::Running(owner) => owner.try_join(),
            ExternalProcessState::Returned(source) => source.try_join(),
            ExternalProcessState::Failed => {
                return Err(SearchSessionFailure::physical_completion_unknown());
            }
        };
        if let Some(joined) = joined {
            self.shutdown = Some(
                joined
                    .map_err(|error| SearchSessionFailure::debug("RoventExternalJoin", &error))?,
            );
        }
        Ok(self.shutdown.is_some_and(|receipt| {
            receipt.cleanup_complete && !receipt.quarantined && !receipt.ownership_lost
        }))
    }

    pub(super) fn shutdown_receipt(&self) -> Option<CheckerShutdown> {
        self.shutdown
    }
}

impl RoventExternalSessionSource<RoventInputLock> {
    pub(super) fn retained_root_admission(&self) -> Option<&Arc<RoventRootAdmission>> {
        match &self.state {
            ExternalProcessState::Returned(source) => Some(source.root_admission()),
            ExternalProcessState::Running(owner) => owner.actual_root_admission(),
            _ => None,
        }
    }
}

impl RoventExternalSessionSource<RoventInputLockV2> {
    pub(super) fn same_original_lock(&self, original: &Arc<RoventInputLockV2>) -> bool {
        match &self.state {
            ExternalProcessState::Returned(source) => source
                .root_admission()
                .owner()
                .v2()
                .is_some_and(|owner| Arc::ptr_eq(owner, original)),
            _ => self
                .custody
                .as_ref()
                .is_some_and(|custody| Arc::ptr_eq(custody.original_lock_owner(), original)),
        }
    }

    pub(super) fn ready_for_root_policy_registration(&self) -> bool {
        matches!(self.state, ExternalProcessState::Dormant(_))
            && self.custody.is_some()
            && self.shutdown.is_none()
    }

    pub(super) fn permits_initial_idle(&self) -> bool {
        match &self.state {
            ExternalProcessState::Dormant(_) => self.ready_for_root_policy_registration(),
            ExternalProcessState::Returned(source) => {
                source.root_drained() && !source.is_quarantined()
            }
            _ => false,
        }
    }

    pub(super) fn retained_root_admission_v2(
        &self,
    ) -> Option<&Arc<crate::rovent_root_admission::source_seal::RoventRootAdmissionV2>> {
        match &self.state {
            ExternalProcessState::Returned(source) => Some(source.root_admission()),
            ExternalProcessState::Running(owner) => owner.actual_root_admission_v2(),
            _ => None,
        }
    }
}

impl<B> AdaptiveSessionDriver<B>
where
    B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
    B::Output: ProposerRankingOutput,
{
    /// /2 is admitted only with its retained pinned researcher and /2 root
    /// policy. Collection needs its own original-owner join and stays closed.
    #[allow(clippy::too_many_arguments, clippy::result_large_err)]
    pub(crate) fn new_rovent_external_v2_with_resource_limits(
        options: AdaptiveSessionOptions,
        limits: AdaptiveExecutionLimits,
        sources: RoventExternalSessionSourcesV2<B>,
        clock: ProcessClock,
        issuer: AdaptiveRulesOwnerIssuer,
        registration: PalsCheckerRegistration,
        profile: RoventPcrSessionProfile,
        retirement_sink: RegisteredProfiledRetirementSink,
        origin: Option<crate::pals_adaptive::RoventNativeCollectionOrigin>,
        hooks: AdaptiveSessionCollectionHooks<B>,
    ) -> Result<Self, RoventExternalSessionAdmissionRejectionV2<B>> {
        let RoventExternalSessionSourcesV2 { common, external } = sources;
        let checked = external
            .validate_declaration(&profile, &common)
            .and_then(|()| {
                if retirement_sink.admission_closed() {
                    return Err(failure(
                        "RoventExternalRetirementSink",
                        "original durable evidence sink admission is closed",
                    ));
                }
                if origin.is_some() || hooks.root_admission.is_some() {
                    return Err(failure(
                        "RoventExternalCollectionHooks",
                        "/2 external Collection owner is not admitted",
                    ));
                }
                Ok(())
            });
        if let Err(failure) = checked {
            return Err(RoventExternalSessionAdmissionRejectionV2 {
                failure,
                sources: common,
                external,
                retirement_sink,
            });
        }
        match Self::admit_with_researcher_source(
            options,
            limits,
            common,
            clock,
            issuer,
            registration,
            None,
            None,
            hooks,
            true,
            SessionResearcherSource::ProfiledExternalV2(Box::new(external)),
        ) {
            Ok(mut driver) => {
                driver.rovent = Some(profile);
                driver.profiled_retirement_sink = Mutex::new(Some(retirement_sink));
                Ok(driver)
            }
            Err((rejected, SessionResearcherSource::ProfiledExternalV2(external))) => {
                Err(RoventExternalSessionAdmissionRejectionV2 {
                    failure: rejected.failure,
                    sources: rejected.sources,
                    external: *external,
                    retirement_sink,
                })
            }
            Err(_) => unreachable!("typed /2 external source preserved"),
        }
    }

    /// Read-only recovery over the same Store owner. This does not acknowledge
    /// persistence or release the retained Work/view pins. A later durable
    /// consumer must supply its own producer-owned commit receipt.
    pub fn inspect_profiled_retirement<R>(
        &self,
        inspect: impl FnOnce(
            &rz_search::pals::store::PalsStores,
            &rz_search::pals::engine::adaptive_port::ProfiledExternalRootRetirement,
        ) -> Result<R, SearchSessionFailure>,
    ) -> Result<Option<R>, SearchSessionFailure> {
        let state = self.state.try_lock().map_err(|_| {
            failure(
                "RoventRetirementBusy",
                "original session owner is busy or poisoned",
            )
        })?;
        let SessionState::Idle(owners) = &*state else {
            return Err(failure(
                "RoventRetirementUnavailable",
                "original idle Store owner is unavailable",
            ));
        };
        let Some(retirement) = &owners.profiled_retirement else {
            return Ok(None);
        };
        let engine = owners
            .engine
            .as_ref()
            .ok_or_else(SearchSessionFailure::physical_completion_unknown)?;
        retirement
            .validate_owner(engine.stores())
            .map_err(|error| SearchSessionFailure::debug("RoventRetirementStore", &error))?;
        retirement
            .protected_pins()
            .map_err(|error| SearchSessionFailure::debug("RoventRetirementPins", &error))?;
        inspect(engine.stores(), retirement).map(Some)
    }

    /// Separate source capability. The old constructors cannot opt into the
    /// zero-native-worker path by changing a bool or checker registration.
    #[allow(clippy::too_many_arguments, clippy::result_large_err)]
    pub fn new_rovent_external_with_resource_limits(
        options: AdaptiveSessionOptions,
        limits: AdaptiveExecutionLimits,
        sources: RoventExternalSessionSources<B>,
        clock: ProcessClock,
        issuer: AdaptiveRulesOwnerIssuer,
        registration: PalsCheckerRegistration,
        profile: RoventPcrSessionProfile,
        retirement_sink: RegisteredProfiledRetirementSink,
        origin: Option<crate::pals_adaptive::RoventNativeCollectionOrigin>,
        hooks: AdaptiveSessionCollectionHooks<B>,
    ) -> Result<Self, RoventExternalSessionAdmissionRejection<B>> {
        let RoventExternalSessionSources { common, external } = sources;
        let checked = external
            .validate_declaration(&profile, &common)
            .and_then(|()| {
                if retirement_sink.admission_closed() {
                    return Err(failure(
                        "RoventExternalRetirementSink",
                        "original durable evidence sink admission is closed",
                    ));
                }
                if origin.is_some() != hooks.root_admission.is_some() {
                    return Err(failure(
                        "RoventExternalCollectionHooks",
                        "Collection origin and actual root observer must be selected together",
                    ));
                }
                Ok(())
            });
        if let Err(failure) = checked {
            return Err(RoventExternalSessionAdmissionRejection {
                failure,
                sources: common,
                external,
                retirement_sink,
            });
        }
        let admitted = Self::admit_with_researcher_source(
            options,
            limits,
            common,
            clock,
            issuer,
            registration,
            None,
            None,
            hooks,
            true,
            SessionResearcherSource::ProfiledExternal(Box::new(external)),
        );
        match admitted {
            Ok(mut driver) => {
                driver.rovent = Some(profile);
                driver.rovent_collection_origin = origin;
                // The cap is moved only at constructor return or while the
                // same one-slot archive worker owns its retirement payload.
                driver.profiled_retirement_sink = Mutex::new(Some(retirement_sink));
                Ok(driver)
            }
            Err((rejected, SessionResearcherSource::ProfiledExternal(external))) => {
                Err(RoventExternalSessionAdmissionRejection {
                    failure: rejected.failure,
                    sources: rejected.sources,
                    external: *external,
                    retirement_sink,
                })
            }
            Err((
                _,
                SessionResearcherSource::Legacy | SessionResearcherSource::ProfiledExternalV2(_),
            )) => unreachable!("typed source preserved"),
        }
    }
}

impl RoventPcrSessionProfile {
    /// Join the original /2 root to the actual worker before attaching the
    /// existing Store-backed frontier. No Collection authority is synthesized.
    pub(super) fn attach_profiled_external_v2<B>(
        &self,
        manager: &mut SessionManager<B>,
        context: &SearchSessionContext,
        clock: &ProcessClock,
        admission: &RoventPcrRootAdmission,
        joined: &RoventAdmittedExternalRootSourceV2,
    ) -> Result<(), AdaptiveManagerError>
    where
        B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
        B::Output: ProposerRankingOutput,
    {
        context.goal.validate(self.maximum_goal_plies)?;
        let original = admission
            .original_v2
            .as_ref()
            .ok_or(AdaptiveManagerError::Invalid(
                "external R requires original /2 root admission",
            ))?;
        joined
            .validate_current()
            .map_err(|error| AdaptiveManagerError::Semantic(error.to_string()))?;
        let source = joined.worker_source();
        let budget = original
            .descriptor()
            .root_budget_digest()
            .map_err(|error| AdaptiveManagerError::Semantic(error.to_string()))?;
        if !context.accepts()?
            || !Arc::ptr_eq(original, joined.root_admission())
            || source.authority() != context.authority
            || source.authority() != original.authority()
            || budget != admission.budget
            || source.clock().domain != clock.domain()
            || source.clock().origin != clock.runtime_clock().origin()
        {
            return Err(AdaptiveManagerError::Invalid(
                "external R /2 actual root/clock/source differs",
            ));
        }
        let port = manager
            .driver()
            .port()
            .ok_or(AdaptiveManagerError::AuxiliaryPending)?;
        let root_rules = port.root_rules();
        let actual_namespace = port
            .model_namespace()
            .map_err(|error| AdaptiveManagerError::Semantic(error.to_string()))?;
        if actual_namespace != self.model_namespace.as_ref() {
            return Err(AdaptiveManagerError::Invalid(
                "external R port/native namespace differs",
            ));
        }
        let root_moves = Arc::new(
            RootMoveDomain::new(&root_rules, context.root_moves.as_deref())
                .map_err(|error| AdaptiveManagerError::Semantic(error.to_string()))?,
        );
        let registration = FrontierRegistration {
            authority: context.authority,
            mode: self.mode,
            original_deadline: clock.deadline(context.control.deadline)?,
            root_budget: admission.budget,
            goal: context.goal,
            nodes: context.requested_nodes,
            selective_policy: self.selective_policy,
            root_rules,
            root_moves: Arc::clone(&root_moves),
            root_state: port.root_state_id(),
            model_namespace: Arc::clone(&self.model_namespace),
            native_ledger: None,
        };
        manager.driver_mut().attach_rovent_external_frontier(
            registration,
            self.frontier_limits.clone(),
            AdaptiveFrontierStartup {
                mode: self.mode,
                root_budget: admission.budget,
                root_moves: &root_moves,
                native_ledger: None,
            },
            source,
            clock.domain(),
            clock.now()?,
        )?;
        manager
            .driver_mut()
            .register_profiled_external_cleanup_deadline(source, context.shutdown_deadline)?;
        manager.configure_rovent_external_routing(self.mode, admission.budget, 1)?;
        manager.configure_rovent_peer_followups(self.peers)?;
        Ok(())
    }

    pub(super) fn attach_profiled_external<B>(
        &self,
        manager: &mut SessionManager<B>,
        context: &SearchSessionContext,
        clock: &ProcessClock,
        collection_origin: Option<&crate::pals_adaptive::RoventNativeCollectionOrigin>,
        admission: &RoventPcrRootAdmission,
        joined: &RoventAdmittedExternalRootSource,
    ) -> Result<(), AdaptiveManagerError>
    where
        B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
        B::Output: ProposerRankingOutput,
    {
        context.goal.validate(self.maximum_goal_plies)?;
        let original = admission
            .original
            .as_ref()
            .ok_or(AdaptiveManagerError::Invalid(
                "external R requires original root admission",
            ))?;
        joined
            .validate_current()
            .map_err(|error| AdaptiveManagerError::Semantic(error.to_string()))?;
        let source = joined.worker_source();
        if !context.accepts()?
            || !Arc::ptr_eq(original, joined.root_admission())
            || source.authority() != context.authority
            || source.authority() != original.authority()
            || original.root_budget_digest() != admission.budget
            || source.clock().domain != clock.domain()
            || source.clock().origin != clock.runtime_clock().origin()
        {
            return Err(AdaptiveManagerError::Invalid(
                "external R actual root/clock/source differs",
            ));
        }
        let port = manager
            .driver()
            .port()
            .ok_or(AdaptiveManagerError::AuxiliaryPending)?;
        let root_rules = port.root_rules();
        let actual_namespace = port
            .model_namespace()
            .map_err(|error| AdaptiveManagerError::Semantic(error.to_string()))?;
        if actual_namespace != self.model_namespace.as_ref() {
            return Err(AdaptiveManagerError::Invalid(
                "external R port/native namespace differs",
            ));
        }
        let root_moves = Arc::new(
            RootMoveDomain::new(&root_rules, context.root_moves.as_deref())
                .map_err(|error| AdaptiveManagerError::Semantic(error.to_string()))?,
        );
        let registration = FrontierRegistration {
            authority: context.authority,
            mode: self.mode,
            original_deadline: clock.deadline(context.control.deadline)?,
            root_budget: admission.budget,
            goal: context.goal,
            nodes: context.requested_nodes,
            selective_policy: self.selective_policy,
            root_rules,
            root_moves: Arc::clone(&root_moves),
            root_state: port.root_state_id(),
            model_namespace: Arc::clone(&self.model_namespace),
            native_ledger: None,
        };
        manager.driver_mut().attach_rovent_external_frontier(
            registration,
            self.frontier_limits.clone(),
            AdaptiveFrontierStartup {
                mode: self.mode,
                root_budget: admission.budget,
                root_moves: &root_moves,
                native_ledger: None,
            },
            source,
            clock.domain(),
            clock.now()?,
        )?;
        // This is the retained actual Start cleanup, registered before the
        // first question. The port never accepts a second/extended window.
        manager
            .driver_mut()
            .register_profiled_external_cleanup_deadline(source, context.shutdown_deadline)?;
        manager.configure_rovent_external_routing(self.mode, admission.budget, 1)?;
        manager.configure_rovent_peer_followups(self.peers)?;
        if let Some(origin) = collection_origin {
            let frontier = manager
                .driver()
                .port()
                .and_then(|port| port.rovent_frontier())
                .ok_or(AdaptiveManagerError::Invalid(
                    "actual external Collection frontier is absent",
                ))?;
            let collection = crate::pals_adaptive::RoventCollectionRootAdmission::from_attached_external_frontier(
                origin,
                frontier,
                self.input_lock,
                admission.budget,
                self.selective_policy,
                Arc::clone(original),
                joined,
            )?;
            manager.admit_collection_root(collection)?;
        }
        Ok(())
    }
}
