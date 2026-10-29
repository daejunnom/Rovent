//! 독립 Collection CPU_T의 원 pinned launch 소유자 초안이다.
//! P-C 검색에 R을 추가하거나 기존 profiled R guard를 낮추지 않는다.
use super::*;
use crate::rovent_root_admission::RoventRootAdmission;
use rz_runtime::contracts::ContractClock;
use rz_search::external_cpu::collection_teacher::TeacherWorkerRegistration;
use rz_search::external_cpu::collection_teacher::{
    CollectionTeacherWorker, TeacherCompletion, TeacherDiagnostics, TeacherDispatch, TeacherJob,
};
use rz_search::external_cpu::process_scope::CollectionProcessPermit;
use std::sync::Arc;

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
mod quarantine_v2;
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
use quarantine_v2::ParkedTeacherV2;
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
use quarantine_v2::TeacherRecoveryReservationV2;
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
pub use quarantine_v2::{TeacherRecoveryCustodianV2, TeacherRecoveryKeyV2};
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
mod actor_binding_v2;
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
pub use actor_binding_v2::{
    BoundCollectionTeacherActorV2, TeacherActorBindingErrorV2, bind_collection_teacher_actor_v2,
    verify_collection_teacher_pin,
};

pub struct RoventCollectionTeacherOwner {
    launch: PreparedRoventExternalLaunch,
    worker: CollectionTeacherWorker,
    root: Arc<RoventRootAdmission>,
    frozen_actor: Digest,
}
impl PreparedRoventExternalLaunch {
    /// 원 FD와 실제 같은 root cap을 보존한다. 하나의 root가 종료될 때까지
    /// 다른 root로 재태그하지 않는다. actor digest는 상위 frozen actor와 대조한다.
    #[allow(clippy::too_many_arguments)]
    pub fn spawn_collection_teacher_pinned(
        self,
        executable: File,
        working_directory: File,
        collection_owner: u32,
        frozen_actor: Digest,
        root: Arc<RoventRootAdmission>,
        clock: ExternalWorkerClock,
        startup_deadline: Instant,
        cleanup_ceiling: Instant,
        cancel: &AtomicBool,
    ) -> Result<RoventCollectionTeacherOwner, CheckerError> {
        self.spawn_collection_teacher_pinned_internal(
            executable,
            working_directory,
            collection_owner,
            frozen_actor,
            root,
            clock,
            startup_deadline,
            cleanup_ceiling,
            cancel,
            None,
        )
    }

    /// The registered collector supplies a same-root process seat. Public
    /// metadata alone cannot reconstruct this permit or another actor's seat.
    #[allow(clippy::too_many_arguments)]
    pub fn spawn_collection_teacher_pinned_in_process_scope(
        self,
        executable: File,
        working_directory: File,
        collection_owner: u32,
        frozen_actor: Digest,
        root: Arc<RoventRootAdmission>,
        clock: ExternalWorkerClock,
        startup_deadline: Instant,
        cleanup_ceiling: Instant,
        cancel: &AtomicBool,
        permit: CollectionProcessPermit,
    ) -> Result<RoventCollectionTeacherOwner, CheckerError> {
        self.spawn_collection_teacher_pinned_internal(
            executable,
            working_directory,
            collection_owner,
            frozen_actor,
            root,
            clock,
            startup_deadline,
            cleanup_ceiling,
            cancel,
            Some(permit),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn spawn_collection_teacher_pinned_internal(
        self,
        executable: File,
        working_directory: File,
        collection_owner: u32,
        frozen_actor: Digest,
        root: Arc<RoventRootAdmission>,
        clock: ExternalWorkerClock,
        startup_deadline: Instant,
        cleanup_ceiling: Instant,
        cancel: &AtomicBool,
        process_permit: Option<CollectionProcessPermit>,
    ) -> Result<RoventCollectionTeacherOwner, CheckerError> {
        // This path accepts only a /1 root. A /2 launch must never be joined
        // to a different owner's /1 root using digest equality alone.
        if self.original_lock_owner_v2().is_some() {
            return Err(invalid("collection teacher /2 requires a /2 root owner"));
        }
        if cancel.load(Ordering::Acquire) || Instant::now() >= startup_deadline {
            return Err(invalid("collection teacher startup canceled/expired"));
        }
        if self.profile().purpose != ExternalEnginePurpose::Teacher
            || frozen_actor.0 == [0; 32]
            || self.input_lock_sha256.as_deref().map(digest).transpose()?
                != Some(root.startup_lock_digest())
            || !root
                .original_context()
                .accepts()
                .map_err(|_| invalid("collection teacher root revoked"))?
            || clock.domain != root.runtime_clock().domain()
            || clock.origin != root.runtime_clock().origin()
        {
            return Err(invalid(
                "collection teacher differs from original pinned purpose/root/clock",
            ));
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (
                executable,
                working_directory,
                collection_owner,
                frozen_actor,
                root,
                clock,
                cleanup_ceiling,
                process_permit,
            );
            Err(CheckerError::Unsupported(
                "collection teacher requires Linux pinned ELF",
            ))
        }
        #[cfg(target_os = "linux")]
        {
            let metadata = executable
                .metadata()
                .map_err(|_| invalid("collection teacher executable metadata"))?;
            if !metadata.is_file()
                || metadata.len() != self.binary.bytes
                || metadata.len() == 0
                || metadata.len() > 256 * 1024 * 1024
            {
                return Err(invalid("collection teacher pinned executable size"));
            }
            let pin = ExternalLaunchPin::from_owned_descriptors(executable, working_directory)?;
            let registration = TeacherWorkerRegistration {
                collection_owner,
                authority: root.authority(),
                original_deadline: root.hard_deadline(),
                input_lock: root.startup_lock_digest(),
                root_admission: root.digest(),
                root_budget: root.root_budget_digest(),
                frozen_actor,
            };
            // The worker validates this original-root registration against the
            // permit before moving it into its private process startup.
            let worker = match process_permit {
                Some(permit) => CollectionTeacherWorker::spawn_in_process_scope(
                    self.profile.configuration()?,
                    self.profile().engine,
                    registration,
                    clock,
                    startup_deadline,
                    cleanup_ceiling,
                    pin,
                    permit,
                ),
                None => CollectionTeacherWorker::spawn(
                    self.profile.configuration()?,
                    self.profile().engine,
                    registration,
                    clock,
                    startup_deadline,
                    cleanup_ceiling,
                    pin,
                ),
            }?;
            Ok(RoventCollectionTeacherOwner {
                launch: self,
                worker,
                root,
                frozen_actor,
            })
        }
    }
}
impl RoventCollectionTeacherOwner {
    pub fn launch(&self) -> &PreparedRoventExternalLaunch {
        &self.launch
    }
    pub fn root_admission(&self) -> &Arc<RoventRootAdmission> {
        &self.root
    }
    pub fn frozen_actor(&self) -> Digest {
        self.frozen_actor
    }
    pub fn worker(&self) -> &CollectionTeacherWorker {
        &self.worker
    }
    /// 과거 실제 startup 관측이다. 현재 liveness나 label 권한은 아니다.
    pub fn verified_startup(&self) -> Result<Option<VerifiedRoventExternalStartup>, CheckerError> {
        let Some(startup) = self.worker.startup_report() else {
            return Ok(None);
        };
        let observed_uci = startup.result?;
        if observed_uci.name != self.launch.profile().expected_uci_name {
            return Err(invalid("collection teacher actual UCI name"));
        }
        let attempt = startup
            .attempt
            .ok_or(invalid("collection teacher startup attempt absent"))?;
        let external = attempt
            .external
            .ok_or(invalid("collection teacher startup process absent"))?;
        if external.request_id != 0
            || external.partial_report.is_some()
            || !reusable(external.process)
        {
            return Err(invalid("collection teacher startup physical evidence"));
        }
        Ok(Some(VerifiedRoventExternalStartup {
            canonical_sha256: self.launch.canonical_sha256.clone(),
            capability_sha256: self.launch.capability_sha256.clone(),
            input_lock_sha256: self
                .launch
                .input_lock_sha256
                .clone()
                .ok_or(invalid("collection teacher startup lock absent"))?,
            observed_uci,
            process: external
                .process
                .process_identity
                .ok_or(invalid("collection teacher process identity"))?,
        }))
    }
    pub fn submit(&mut self, job: TeacherJob) -> Result<TeacherDispatch, CheckerError> {
        let scope = job.input_scope;
        if !self
            .root
            .original_context()
            .accepts()
            .map_err(|_| invalid("collection teacher root revoked"))?
            || scope.authority != self.root.authority()
            || scope.root_admission != self.root.digest()
            || scope.input_lock != self.root.startup_lock_digest()
            || scope.root_budget != self.root.root_budget_digest()
            || scope.frozen_actor != self.frozen_actor
            || job.original_deadline.clock != self.root.hard_deadline().clock
            || job.original_deadline.at > self.root.hard_deadline().at
        {
            return Err(invalid(
                "collection teacher Job differs from original live root",
            ));
        }
        self.launch
            .require_query(job.constraint, job.root_moves.is_some())?;
        self.worker.submit(job)
    }
    /// 실패 terminal은 원 실패로 보존하며 성공 label로 변환하지 않는다.
    pub fn validate_completion(
        &self,
        completion: &TeacherCompletion,
        dispatch: &TeacherDispatch,
    ) -> Result<(), CheckerError> {
        if !completion.matches_submission(dispatch)
            || dispatch.input_scope().authority != self.root.authority()
            || dispatch.input_scope().root_admission != self.root.digest()
            || dispatch.input_scope().frozen_actor != self.frozen_actor
        {
            return Err(invalid(
                "collection teacher completion lacks original private dispatch",
            ));
        }
        if let Ok(outcome) = completion.result() {
            let report = &outcome.report;
            let declared = self.launch.profile();
            let attempt = completion
                .attempt()
                .ok_or(invalid("collection teacher successful attempt absent"))?;
            let external = attempt
                .external
                .as_ref()
                .ok_or(invalid("collection teacher physical receipt absent"))?;
            if report.profile != declared.engine
                || report.purpose != ExternalEnginePurpose::Teacher
                || report.raw.identity != declared.identity
                || report.raw.observed_uci.name != declared.expected_uci_name
                || report.requested != dispatch.constraint()
                || completion.issued_constraint() != Some(report.issued)
                || completion.issued_go() != Some(report.issued_go.as_str())
                || completion.go_issue() != rz_search::external_cpu::ExternalGoIssue::Issued
                || !report.readiness_barrier_completed
                || report.raw.request_id == 0
                || external.request_id != report.raw.request_id
                || external.partial_report.as_ref() != Some(&report.raw)
                || attempt.elapsed != report.raw.elapsed
                || external.process.process_identity != Some(dispatch.process_identity())
                || !reusable(external.process)
            {
                return Err(invalid(
                    "collection teacher raw/private physical provenance mismatch",
                ));
            }
        }
        Ok(())
    }
    pub fn try_join(&mut self) -> Option<Result<CheckerShutdown, CheckerError>> {
        self.worker.try_join()
    }
    pub fn take_diagnostics_after_join(&mut self) -> Option<TeacherDiagnostics> {
        self.worker.take_diagnostics_after_join()
    }
}

/// A teacher worker bound to the original /2 Collection capability. Its
/// source descriptors and actual attached root stay alive until physical exit.
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
pub struct RoventCollectionTeacherOwnerV2 {
    launch: PreparedRoventExternalLaunch,
    worker: CollectionTeacherWorker,
    admission: Option<crate::pals_adaptive::RoventCollectionRootAdmissionV2>,
    custody: Option<crate::rovent_bootstrap::RoventCollectionTeacherSourceCustody>,
    frozen_actor: Digest,
    source_index: usize,
    bound_source: Option<BoundTeacherOriginalSourceV2>,
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
struct BoundTeacherOriginalSourceV2 {
    _manifest: Arc<rz_experiments::rovent_corpus_v2::RoventCorpusManifestV2>,
    _actor_id: String,
    _manifest_sha256: String,
    _profile_bytes: Vec<u8>,
    _executable: File,
    _working_directory: File,
}

/// A post-spawn rejection retains the actual worker and its original source
/// so the caller can request shutdown and collect physical drain evidence.
/// The caller retains the borrowed permit in both variants.
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
pub enum RoventCollectionTeacherSpawnRejectionV2 {
    BeforeSpawn {
        cause: CheckerError,
        admission: crate::pals_adaptive::RoventCollectionRootAdmissionV2,
    },
    AfterSpawn {
        cause: CheckerError,
        owner: RoventCollectionTeacherOwnerV2,
    },
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
impl RoventCollectionTeacherOwnerV2 {
    /// Obtain the source bundle inside this method from the opaque real root.
    /// A caller cannot supply a replacement executable or cwd. The caller must
    /// separately bind `source_index` and `frozen_actor` to its frozen corpus
    /// registration; the permit alone is not teacher-label authority. The
    /// borrowed permit remains with its caller if pre-spawn admission fails.
    #[allow(clippy::too_many_arguments)]
    // Not exported for product use until the session wires root revoke to
    // worker shutdown and preserves the subsequent physical drain evidence.
    pub(crate) fn spawn(
        admission: crate::pals_adaptive::RoventCollectionRootAdmissionV2,
        source_index: usize,
        control: &crate::rovent_bootstrap::RoventStartupControl,
        collection_owner: u32,
        frozen_actor: Digest,
        clock: ExternalWorkerClock,
        startup_deadline: Instant,
        cleanup_ceiling: Instant,
        cancel: &AtomicBool,
        permit: &CollectionProcessPermit,
        scope: &crate::pals_adaptive::RoventTeacherScopeCapabilityV2,
    ) -> Result<Self, RoventCollectionTeacherSpawnRejectionV2> {
        let _scope_guard = match scope.begin_spawn(&admission, permit) {
            Ok(guard) => guard,
            Err(_) => return Err(RoventCollectionTeacherSpawnRejectionV2::BeforeSpawn {
                cause: invalid("teacher /2 scope invalid, closed, or contended"),
                admission,
            }),
        };
        let created: Result<Self, CheckerError> = (|| {
            let root = admission.root_admission();
            let original = root
                .owner()
                .v2()
                .ok_or(invalid("teacher /2 lock missing"))?;
            let preload = admission.native_origin().collection_preload();
            let original_preload = admission
                .native_origin()
                .sources()
                .verify_collection_v2_origin()
                .map_err(|_| invalid("teacher /2 original preload"))?;
            let admission_limit = clock
                .origin
                .checked_add(std::time::Duration::from_nanos(
                    root.admission_deadline().at.0,
                ))
                .ok_or(invalid("teacher /2 admission deadline overflow"))?;
            let root_cleanup_limit = clock
                .origin
                .checked_add(std::time::Duration::from_nanos(
                    root.cleanup_deadline().at.0,
                ))
                .ok_or(invalid("teacher /2 cleanup deadline overflow"))?;
            if !Arc::ptr_eq(preload, original_preload)
                || !preload.matches_input_lock_v2(original)
                || !Arc::ptr_eq(
                    &control.cancellation,
                    &root.original_context().control.cancellation,
                )
                || !std::ptr::eq(
                    Arc::as_ptr(&control.cancellation),
                    cancel as *const AtomicBool,
                )
                || control.until > admission_limit
                || control.cleanup_until > root_cleanup_limit
                || startup_deadline > control.until
                || cleanup_ceiling > control.cleanup_until
                || !root
                    .original_context()
                    .try_accepts()
                    .map_err(|_| invalid("teacher /2 root revoked"))?
                || frozen_actor == Digest([0; 32])
                || clock.domain != root.runtime_clock().domain()
                || clock.origin != root.runtime_clock().origin()
                || cancel.load(Ordering::Acquire)
                || Instant::now() >= startup_deadline
            {
                return Err(invalid("teacher /2 source/root/control/clock mismatch"));
            }
            root.descriptor()
                .verify_against(original)
                .map_err(|_| invalid("teacher /2 descriptor"))?;
            let root_digest = root
                .descriptor()
                .digest()
                .map_err(|_| invalid("teacher /2 root digest"))?;
            let root_budget = root
                .descriptor()
                .root_budget_digest()
                .map_err(|_| invalid("teacher /2 budget digest"))?;
            if root_budget != admission.root_budget() {
                return Err(invalid("teacher /2 budget mismatch"));
            }
            let sources = preload
                .teacher_sources(source_index, control)
                .map_err(|_| invalid("teacher /2 source custody"))?;
            let (index, launch, _profile, executable, working_directory, custody) =
                sources.into_parts();
            if index != source_index
                || !custody.matches_preload(preload)
                || !launch
                    .original_lock_owner_v2()
                    .is_some_and(|owner| Arc::ptr_eq(owner, original))
                || launch.profile().purpose != ExternalEnginePurpose::Teacher
                || launch
                    .input_lock_sha256
                    .as_deref()
                    .map(digest)
                    .transpose()?
                    != Some(root.startup_lock_digest())
            {
                return Err(invalid("teacher /2 original source owner mismatch"));
            }
            #[cfg(not(target_os = "linux"))]
            {
                let _ = (
                    executable,
                    working_directory,
                    collection_owner,
                    clock,
                    cleanup_ceiling,
                    permit,
                    custody,
                    root_digest,
                );
                Err(CheckerError::Unsupported(
                    "collection teacher requires Linux pinned ELF",
                ))
            }
            #[cfg(target_os = "linux")]
            {
                let root_cleanup = clock
                    .origin
                    .checked_add(std::time::Duration::from_nanos(
                        root.cleanup_deadline().at.0,
                    ))
                    .ok_or(invalid("teacher /2 cleanup deadline overflow"))?;
                let hard = clock
                    .origin
                    .checked_add(std::time::Duration::from_nanos(root.hard_deadline().at.0))
                    .ok_or(invalid("teacher /2 hard deadline overflow"))?;
                let config = launch.configuration()?;
                let profile_cleanup = hard
                    .checked_add(config.shutdown_grace)
                    .ok_or(invalid("teacher /2 profile cleanup overflow"))?;
                let cleanup_ceiling = cleanup_ceiling.min(root_cleanup).min(profile_cleanup);
                let pin = ExternalLaunchPin::from_owned_descriptors(executable, working_directory)?;
                let registration = TeacherWorkerRegistration {
                    collection_owner,
                    authority: root.authority(),
                    original_deadline: root.hard_deadline(),
                    input_lock: root.startup_lock_digest(),
                    root_admission: root_digest,
                    root_budget,
                    frozen_actor,
                };
                if !root
                    .original_context()
                    .try_accepts()
                    .map_err(|_| invalid("teacher /2 root revoked"))?
                    || cancel.load(Ordering::Acquire)
                {
                    return Err(invalid("teacher /2 revoked before worker startup"));
                }
                let worker = CollectionTeacherWorker::spawn_in_process_scope(
                    config,
                    launch.profile().engine,
                    registration,
                    clock,
                    startup_deadline,
                    cleanup_ceiling,
                    pin,
                    permit.clone(),
                )?;
                let owner = Self {
                    launch,
                    worker,
                    admission: Some(admission.clone()),
                    custody: Some(custody),
                    frozen_actor,
                    source_index,
                    bound_source: None,
                };
                Ok(owner)
            }
        })();
        let owner = match created {
            Ok(owner) => owner,
            Err(cause) => {
                return Err(RoventCollectionTeacherSpawnRejectionV2::BeforeSpawn {
                    cause,
                    admission,
                });
            }
        };
        let validity = owner
            .admission()
            .root_admission()
            .original_context()
            .try_accepts();
        let cause = match validity {
            Ok(true) if !cancel.load(Ordering::Acquire) => return Ok(owner),
            Ok(_) => invalid("teacher /2 revoked during worker startup"),
            Err(_) => invalid("teacher /2 root authority unavailable after worker startup"),
        };
        owner.worker.request_shutdown(Instant::now());
        Err(RoventCollectionTeacherSpawnRejectionV2::AfterSpawn { cause, owner })
    }

    pub fn worker(&self) -> &CollectionTeacherWorker {
        &self.worker
    }
    pub fn source_index(&self) -> usize {
        self.source_index
    }
    pub fn launch(&self) -> &PreparedRoventExternalLaunch {
        &self.launch
    }
    pub fn admission(&self) -> &crate::pals_adaptive::RoventCollectionRootAdmissionV2 {
        self.admission.as_ref().expect("live teacher /2 admission")
    }
    pub fn verified_startup(&self) -> Result<Option<VerifiedRoventExternalStartup>, CheckerError> {
        let Some(startup) = self.worker.startup_report() else {
            return Ok(None);
        };
        let observed_uci = startup.result?;
        if observed_uci.name != self.launch.profile().expected_uci_name {
            return Err(invalid("collection teacher actual UCI name"));
        }
        let attempt = startup
            .attempt
            .ok_or(invalid("collection teacher startup attempt absent"))?;
        let external = attempt
            .external
            .ok_or(invalid("collection teacher startup process absent"))?;
        if external.request_id != 0
            || external.partial_report.is_some()
            || !reusable(external.process)
        {
            return Err(invalid("collection teacher startup physical evidence"));
        }
        Ok(Some(VerifiedRoventExternalStartup {
            canonical_sha256: self.launch.canonical_sha256.clone(),
            capability_sha256: self.launch.capability_sha256.clone(),
            input_lock_sha256: self
                .launch
                .input_lock_sha256
                .clone()
                .ok_or(invalid("collection teacher startup lock absent"))?,
            observed_uci,
            process: external
                .process
                .process_identity
                .ok_or(invalid("collection teacher process identity"))?,
        }))
    }
    pub fn submit(&mut self, job: TeacherJob) -> Result<TeacherDispatch, CheckerError> {
        let root = self.admission().root_admission();
        let scope = job.input_scope;
        if !root
            .original_context()
            .accepts()
            .map_err(|_| invalid("teacher /2 root revoked"))?
            || scope.authority != root.authority()
            || scope.root_admission
                != root
                    .descriptor()
                    .digest()
                    .map_err(|_| invalid("teacher /2 root digest"))?
            || scope.input_lock != root.startup_lock_digest()
            || scope.root_budget != self.admission().root_budget()
            || scope.frozen_actor != self.frozen_actor
            || job.original_deadline.clock != root.hard_deadline().clock
            || job.original_deadline.at > root.hard_deadline().at
        {
            return Err(invalid("teacher /2 job differs from original live root"));
        }
        self.launch
            .require_query(job.constraint, job.root_moves.is_some())?;
        self.worker.submit(job)
    }
    pub fn validate_completion(
        &self,
        completion: &TeacherCompletion,
        dispatch: &TeacherDispatch,
    ) -> Result<(), CheckerError> {
        if !self
            .admission()
            .root_admission()
            .original_context()
            .accepts()
            .map_err(|_| invalid("teacher /2 root revoked"))?
            || !completion.matches_submission(dispatch)
            || dispatch.input_scope().input_lock
                != self.admission().root_admission().startup_lock_digest()
            || dispatch.input_scope().root_budget != self.admission().root_budget()
            || dispatch.input_scope().authority != self.admission().root_admission().authority()
            || dispatch.input_scope().root_admission
                != self
                    .admission()
                    .root_admission()
                    .descriptor()
                    .digest()
                    .map_err(|_| invalid("teacher /2 root digest"))?
            || dispatch.input_scope().frozen_actor != self.frozen_actor
        {
            return Err(invalid(
                "collection teacher completion lacks original private dispatch",
            ));
        }
        if let Ok(outcome) = completion.result() {
            let report = &outcome.report;
            let declared = self.launch.profile();
            let attempt = completion
                .attempt()
                .ok_or(invalid("collection teacher successful attempt absent"))?;
            let external = attempt
                .external
                .as_ref()
                .ok_or(invalid("collection teacher physical receipt absent"))?;
            if report.profile != declared.engine
                || report.purpose != ExternalEnginePurpose::Teacher
                || report.raw.identity != declared.identity
                || report.raw.observed_uci.name != declared.expected_uci_name
                || report.requested != dispatch.constraint()
                || completion.issued_constraint() != Some(report.issued)
                || completion.issued_go() != Some(report.issued_go.as_str())
                || completion.go_issue() != rz_search::external_cpu::ExternalGoIssue::Issued
                || !report.readiness_barrier_completed
                || report.raw.request_id == 0
                || external.request_id != report.raw.request_id
                || external.partial_report.as_ref() != Some(&report.raw)
                || attempt.elapsed != report.raw.elapsed
                || external.process.process_identity != Some(dispatch.process_identity())
                || !reusable(external.process)
            {
                return Err(invalid(
                    "collection teacher raw/private physical provenance mismatch",
                ));
            }
        }
        Ok(())
    }
    pub fn try_join(&mut self) -> Option<Result<CheckerShutdown, CheckerError>> {
        self.worker.try_join()
    }
}

/// The original declaration-bound actor is returned on every pre-spawn error.
/// Once a worker exists, its lifecycle and recovery key are returned instead.
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
pub enum BoundTeacherSpawnFailureV2 {
    BeforeSpawn {
        cause: BoundTeacherSpawnCauseV2,
        actor: BoundCollectionTeacherActorV2,
    },
    AfterSpawn {
        cause: BoundTeacherSpawnCauseV2,
        lifecycle: RoventCollectionTeacherLifecycleV2,
        recovery_key: TeacherRecoveryKeyV2,
    },
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
#[derive(Debug)]
pub enum BoundTeacherSpawnCauseV2 {
    Root(rz_contracts::ContractError),
    Scope(crate::pals_adaptive::AdaptiveCollectionError),
    Source(rz_eval::error::BackendError),
    Io(std::io::Error),
    Checker(CheckerError),
}

/// Keeps the original actor and worker under the pre-reserved recovery slot
/// during the short handoff to the lifecycle. This only covers unwinding
/// after `spawn_in_process_scope` returns a worker; it cannot recover from an
/// abort or from an unobserved failure inside the lower spawn implementation.
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
struct SpawnedTeacherGuardV2 {
    actor: Option<BoundCollectionTeacherActorV2>,
    worker: Option<CollectionTeacherWorker>,
    reservation: Option<TeacherRecoveryReservationV2>,
    cleanup_deadline: Instant,
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
fn owner_from_spawned_actor_v2(
    actor: BoundCollectionTeacherActorV2,
    worker: CollectionTeacherWorker,
) -> RoventCollectionTeacherOwnerV2 {
    // Both conversions below only move the original fields. They must remain
    // allocation-free and non-fallible while the worker has no other owner.
    let parts = actor.into_parts();
    let (index, launch, profile_bytes, executable, working_directory, custody) =
        parts.source.into_parts();
    RoventCollectionTeacherOwnerV2 {
        launch,
        worker,
        admission: Some(parts.admission),
        custody: Some(custody),
        frozen_actor: parts.actor_digest,
        source_index: index,
        bound_source: Some(BoundTeacherOriginalSourceV2 {
            _manifest: parts.manifest,
            _actor_id: parts.actor_id,
            _manifest_sha256: parts.manifest_sha256,
            _profile_bytes: profile_bytes,
            _executable: executable,
            _working_directory: working_directory,
        }),
    }
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
impl SpawnedTeacherGuardV2 {
    fn into_lifecycle(mut self) -> RoventCollectionTeacherLifecycleV2 {
        let actor = self.actor.take().expect("spawn guard retains actor");
        let worker = self.worker.take().expect("spawn guard retains worker");
        let reservation = self
            .reservation
            .take()
            .expect("spawn guard retains recovery reservation");
        let owner = owner_from_spawned_actor_v2(actor, worker);
        RoventCollectionTeacherLifecycleV2::new(owner, reservation, self.cleanup_deadline)
    }
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
impl Drop for SpawnedTeacherGuardV2 {
    fn drop(&mut self) {
        if self.actor.is_none() && self.worker.is_none() && self.reservation.is_none() {
            return;
        }
        if self.actor.is_none() || self.worker.is_none() || self.reservation.is_none() {
            // An impossible partial handoff must not silently destroy a
            // remaining worker or release a possibly occupied reservation.
            quarantine_v2::mark_custody_violation();
            if let Some(actor) = self.actor.take() {
                std::mem::forget(actor);
            }
            if let Some(worker) = self.worker.take() {
                std::mem::forget(worker);
            }
            if let Some(reservation) = self.reservation.take() {
                std::mem::forget(reservation);
            }
            return;
        }
        if let (Some(actor), Some(worker), Some(reservation)) = (
            self.actor.take(),
            self.worker.take(),
            self.reservation.take(),
        ) {
            let owner = owner_from_spawned_actor_v2(actor, worker);
            // Lifecycle Drop requests shutdown and parks the same owner in
            // the already registered recovery slot; it never claims exit.
            drop(RoventCollectionTeacherLifecycleV2::new(
                owner,
                reservation,
                self.cleanup_deadline,
            ));
        }
    }
}

/// Starts only the actor bound to the original /2 root and preload. It does
/// not grant label or publication authority. The caller retains the borrowed
/// process permit and must retain the returned recovery key beyond the session.
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
pub fn spawn_bound_collection_teacher_v2(
    actor: BoundCollectionTeacherActorV2,
    permit: &CollectionProcessPermit,
    scope: &crate::pals_adaptive::RoventTeacherScopeCapabilityV2,
    startup_deadline: Instant,
    cleanup_ceiling: Instant,
    custodian: &mut TeacherRecoveryCustodianV2,
) -> Result<(RoventCollectionTeacherLifecycleV2, TeacherRecoveryKeyV2), BoundTeacherSpawnFailureV2>
{
    // Serialize this complete spawn attempt with scope close. A stale root or
    // another root's capability cannot start a worker even with a separately
    // fabricated process permit. This guard is released only on return.
    let _scope_guard = match scope.begin_spawn(actor.admission(), permit) {
        Ok(guard) => guard,
        Err(error) => return Err(BoundTeacherSpawnFailureV2::BeforeSpawn {
            cause: BoundTeacherSpawnCauseV2::Scope(error),
            actor,
        }),
    };
    let (reservation, recovery_key) = match TeacherRecoveryReservationV2::reserve() {
        Ok(value) => value,
        Err(cause) => {
            return Err(BoundTeacherSpawnFailureV2::BeforeSpawn {
                cause: BoundTeacherSpawnCauseV2::Checker(cause),
                actor,
            });
        }
    };
    if let Err(cause) = custodian.register(&recovery_key) {
        return Err(BoundTeacherSpawnFailureV2::BeforeSpawn {
            cause: BoundTeacherSpawnCauseV2::Checker(cause),
            actor,
        });
    }
    let cleanup_deadline = actor.admission().root_admission().original_context().shutdown_deadline;
    let created = (|| -> Result<CollectionTeacherWorker, BoundTeacherSpawnCauseV2> {
        reservation
            .validate_before_spawn()
            .map_err(BoundTeacherSpawnCauseV2::Checker)?;
        let admission = actor.admission();
        let root = admission.root_admission();
        let control = actor.root_control();
        let context = root.original_context();
        let source = actor.source();
        let lock = root
            .owner()
            .v2()
            .ok_or(BoundTeacherSpawnCauseV2::Checker(invalid(
                "teacher /2 original lock absent",
            )))?;
        if !Arc::ptr_eq(&control.cancellation, &context.control.cancellation)
            || !source
                .prepared
                .original_lock_owner_v2()
                .is_some_and(|owner| Arc::ptr_eq(owner, lock))
            || source.index != actor.source_index()
            || control.cleanup_until > context.shutdown_deadline
            || startup_deadline > control.until
            || cleanup_ceiling > control.cleanup_until
            || actor.actor_digest() == Digest([0; 32])
        {
            return Err(BoundTeacherSpawnCauseV2::Checker(invalid(
                "teacher /2 original source, root or deadline mismatch",
            )));
        }
        control.check().map_err(BoundTeacherSpawnCauseV2::Source)?;
        if !context
            .try_accepts()
            .map_err(BoundTeacherSpawnCauseV2::Root)?
        {
            return Err(BoundTeacherSpawnCauseV2::Checker(invalid(
                "teacher /2 root revoked",
            )));
        }
        let root_digest = root
            .descriptor()
            .digest()
            .map_err(|_| BoundTeacherSpawnCauseV2::Checker(invalid("teacher /2 root digest")))?;
        let registration = TeacherWorkerRegistration {
            collection_owner: actor.collection_owner(),
            authority: root.authority(),
            original_deadline: root.hard_deadline(),
            input_lock: root.startup_lock_digest(),
            root_admission: root_digest,
            root_budget: admission.root_budget(),
            frozen_actor: actor.actor_digest(),
        };
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (permit, registration);
            Err(BoundTeacherSpawnCauseV2::Checker(
                CheckerError::Unsupported("collection teacher requires Linux pinned ELF"),
            ))
        }
        #[cfg(target_os = "linux")]
        {
            let executable = source
                .executable
                .try_clone()
                .map_err(BoundTeacherSpawnCauseV2::Io)?;
            let directory = source
                .working_directory
                .try_clone()
                .map_err(BoundTeacherSpawnCauseV2::Io)?;
            let pin = ExternalLaunchPin::from_owned_descriptors(executable, directory)
                .map_err(BoundTeacherSpawnCauseV2::Checker)?;
            let config = source
                .prepared
                .configuration()
                .map_err(BoundTeacherSpawnCauseV2::Checker)?;
            let clock = ExternalWorkerClock {
                domain: root.runtime_clock().domain(),
                origin: root.runtime_clock().origin(),
            };
            let hard = clock
                .origin
                .checked_add(std::time::Duration::from_nanos(root.hard_deadline().at.0))
                .ok_or(BoundTeacherSpawnCauseV2::Checker(invalid(
                    "teacher /2 hard deadline overflow",
                )))?;
            let profile_cleanup = hard.checked_add(config.shutdown_grace).ok_or(
                BoundTeacherSpawnCauseV2::Checker(invalid("teacher /2 profile cleanup overflow")),
            )?;
            let ceiling = cleanup_ceiling.min(profile_cleanup);
            if !context
                .try_accepts()
                .map_err(BoundTeacherSpawnCauseV2::Root)?
            {
                return Err(BoundTeacherSpawnCauseV2::Checker(invalid(
                    "teacher /2 root revoked before worker startup",
                )));
            }
            reservation
                .validate_before_spawn()
                .map_err(BoundTeacherSpawnCauseV2::Checker)?;
            CollectionTeacherWorker::spawn_in_process_scope(
                config,
                source.prepared.profile().engine,
                registration,
                clock,
                startup_deadline,
                ceiling,
                pin,
                permit.clone(),
            )
            .map_err(BoundTeacherSpawnCauseV2::Checker)
        }
    })();
    let worker = match created {
        Ok(worker) => worker,
        Err(cause) => return Err(BoundTeacherSpawnFailureV2::BeforeSpawn { cause, actor }),
    };
    // No fallible operation occurs between the returned worker and this guard.
    let mut lifecycle = SpawnedTeacherGuardV2 {
        actor: Some(actor),
        worker: Some(worker),
        reservation: Some(reservation),
        cleanup_deadline,
    }
    .into_lifecycle();
    let root = lifecycle
        .owner
        .as_ref()
        .expect("retained teacher owner")
        .admission()
        .root_admission();
    let observation = root.original_context().try_accepts();
    match observation {
        Ok(true) => Ok((lifecycle, recovery_key)),
        Ok(false) => {
            lifecycle.request_shutdown(cleanup_ceiling);
            Err(BoundTeacherSpawnFailureV2::AfterSpawn {
                cause: BoundTeacherSpawnCauseV2::Checker(invalid(
                    "teacher /2 root revoked during worker startup",
                )),
                lifecycle,
                recovery_key,
            })
        }
        Err(error) => {
            lifecycle.root_observation_error = Some(error);
            // A contended root lock is not root revocation. Preserve the
            // already-started owner so the caller can retry observation.
            if error.code != rz_contracts::ErrorCode::ResourceExhausted {
                lifecycle.request_shutdown(cleanup_ceiling);
            }
            Err(BoundTeacherSpawnFailureV2::AfterSpawn {
                cause: BoundTeacherSpawnCauseV2::Root(error),
                lifecycle,
                recovery_key,
            })
        }
    }
}

/// One observed retirement of the original teacher process. Even Confirmed
/// does not mean its queued completions were recovered or its parent process
/// scope closed. It is not root revocation, a native fence, or label authority.
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
#[derive(Clone, Debug)]
pub enum RoventTeacherPhysicalRetirementV2 {
    NotRequested,
    Pending,
    Confirmed(CheckerShutdown),
    Unknown(Option<CheckerShutdown>),
}

/// Whether the original diagnostics were returned to the caller. Unavailable
/// does not distinguish unpublished bytes from bytes taken by an earlier owner.
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoventTeacherDiagnosticsDispositionV2 {
    NotObserved,
    Recovered,
    UnavailableOrAlreadyTaken,
}

/// Errors, actual process evidence and undrained work remain separate. A
/// confirmed process retirement does not imply that its output was consumed.
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
#[derive(Clone, Debug)]
pub struct RoventTeacherRetirementObservationV2 {
    pub root_observation_error: Option<rz_contracts::ContractError>,
    pub worker_join: Option<Result<CheckerShutdown, CheckerError>>,
    pub physical: RoventTeacherPhysicalRetirementV2,
    pub commands: usize,
    pub reliable_queued: usize,
    pub reliable_reserved: usize,
    pub diagnostics: RoventTeacherDiagnosticsDispositionV2,
    /// This wrapper tracks actual terminal delivery for its own accepted
    /// dispatches. It does not track label consumption or publication.
    pub terminal_accounting: RoventTeacherTerminalAccountingV2,
    /// Recovery of a poisoned registry was needed; do not infer its health.
    pub registry_poisoned: bool,
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoventTeacherTerminalAccountingV2 {
    /// Every accepted dispatch has had its matching terminal delivered to the
    /// caller. This is not a statement that the caller consumed a label.
    Delivered { submitted: u64, delivered: u64 },
    /// A submitted dispatch still awaits a matching terminal.
    Pending { submitted: u64, delivered: u64 },
    /// A missing or mismatched terminal prevents a complete accounting.
    Unknown,
}

/// Proof that this exact actor owner passed `try_finish`'s physical and
/// terminal checks. It is not a roster-wide proof, a successful worker result,
/// or authority to consume a teacher label or publish a corpus row.
/// The fields are private so a caller cannot promote a copied observation to
/// physical retirement evidence.
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
pub struct RoventTeacherRetirementTokenV2 {
    owner: RoventCollectionTeacherOwnerV2,
    observation: RoventTeacherRetirementObservationV2,
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
impl RoventTeacherRetirementTokenV2 {
    pub fn matches_admission(
        &self,
        admission: &crate::pals_adaptive::RoventCollectionRootAdmissionV2,
    ) -> bool {
        self.owner.admission().same_original_admission(admission)
    }

    pub fn owner(&self) -> &RoventCollectionTeacherOwnerV2 {
        &self.owner
    }

    pub fn observation(&self) -> &RoventTeacherRetirementObservationV2 {
        &self.observation
    }
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
#[derive(Default)]
struct TeacherTerminalLedgerV2 {
    pending: Option<TeacherDispatch>,
    unmatched: Option<TeacherCompletion>,
    submitted: u64,
    delivered: u64,
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
impl TeacherTerminalLedgerV2 {
    fn observation(&self) -> RoventTeacherTerminalAccountingV2 {
        if self.unmatched.is_some() || self.delivered > self.submitted {
            RoventTeacherTerminalAccountingV2::Unknown
        } else if self.pending.is_some() {
            RoventTeacherTerminalAccountingV2::Pending {
                submitted: self.submitted,
                delivered: self.delivered,
            }
        } else if self.submitted == self.delivered {
            RoventTeacherTerminalAccountingV2::Delivered {
                submitted: self.submitted,
                delivered: self.delivered,
            }
        } else {
            RoventTeacherTerminalAccountingV2::Unknown
        }
    }
}

/// Owns the original /2 teacher across admission closure, shutdown and join.
/// The surrounding session still owns root revoke and the process scope. It
/// must call `observe_root` as part of its control loop; this type starts no
/// watcher and does not independently authorize Collection execution.
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
pub struct RoventCollectionTeacherLifecycleV2 {
    owner: Option<RoventCollectionTeacherOwnerV2>,
    reservation: Option<TeacherRecoveryReservationV2>,
    shutdown_requested: bool,
    root_observation_error: Option<rz_contracts::ContractError>,
    diagnostics: RoventTeacherDiagnosticsDispositionV2,
    cleanup_deadline: Instant,
    registry_poisoned: bool,
    terminal_ledger: TeacherTerminalLedgerV2,
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
impl RoventCollectionTeacherLifecycleV2 {
    /// The retained original owner, without transferring lifecycle custody.
    /// A borrowed owner by itself is not a dispatch or retirement proof.
    pub fn owner(&self) -> &RoventCollectionTeacherOwnerV2 {
        self.owner.as_ref().expect("retained teacher owner")
    }

    pub(crate) fn new(
        owner: RoventCollectionTeacherOwnerV2,
        reservation: TeacherRecoveryReservationV2,
        cleanup_deadline: Instant,
    ) -> Self {
        Self {
            owner: Some(owner),
            reservation: Some(reservation),
            shutdown_requested: false,
            root_observation_error: None,
            diagnostics: RoventTeacherDiagnosticsDispositionV2::NotObserved,
            cleanup_deadline,
            registry_poisoned: false,
            terminal_ledger: TeacherTerminalLedgerV2::default(),
        }
    }

    fn from_parked(reservation: TeacherRecoveryReservationV2, parked: ParkedTeacherV2) -> Self {
        Self {
            owner: Some(parked.owner),
            reservation: Some(reservation),
            shutdown_requested: parked.shutdown_requested,
            root_observation_error: parked.root_observation_error,
            diagnostics: parked.diagnostics,
            cleanup_deadline: parked.cleanup_deadline,
            registry_poisoned: parked.registry_poisoned,
            terminal_ledger: parked.terminal_ledger,
        }
    }

    /// Observe the original root without waiting for its mutex. Contention is
    /// an unavailable observation, not proof that the root was revoked.
    pub fn observe_root(&mut self) -> Result<bool, rz_contracts::ContractError> {
        if self.shutdown_requested {
            return Ok(false);
        }
        let context = self
            .owner
            .as_ref()
            .expect("retained teacher owner")
            .admission()
            .root_admission()
            .original_context();
        let until = context.shutdown_deadline;
        match context.try_accepts() {
            Ok(true) => Ok(true),
            Ok(false) => {
                self.request_shutdown(until);
                Ok(false)
            }
            Err(error) => {
                self.root_observation_error = Some(error);
                if error.code != rz_contracts::ErrorCode::ResourceExhausted {
                    self.request_shutdown(until);
                }
                Err(error)
            }
        }
    }

    /// The lower worker closes its mailbox and bounds the deadline by the
    /// original cleanup ceiling. A request is not evidence of physical exit.
    pub fn request_shutdown(&mut self, until: Instant) {
        self.shutdown_requested = true;
        self.owner
            .as_ref()
            .expect("retained teacher owner")
            .worker
            .request_shutdown(until);
    }

    /// Submit through the original live root and bound worker. The returned
    /// dispatch is not a label or publication authority.
    pub fn submit(&mut self, job: TeacherJob) -> Result<TeacherDispatch, CheckerError> {
        if self.terminal_ledger.pending.is_some() || self.terminal_ledger.unmatched.is_some() {
            return Err(invalid("teacher /2 previous terminal unresolved"));
        }
        let next_submitted = self
            .terminal_ledger
            .submitted
            .checked_add(1)
            .ok_or_else(|| invalid("teacher /2 dispatch accounting overflow"))?;
        if !self
            .observe_root()
            .map_err(|_| invalid("teacher /2 root observation unavailable"))?
        {
            return Err(invalid("teacher /2 admission closed"));
        }
        let result = self
            .owner
            .as_mut()
            .expect("retained teacher owner")
            .submit(job);
        if let Ok(dispatch) = &result {
            self.terminal_ledger.pending = Some(dispatch.clone());
            self.terminal_ledger.submitted = next_submitted;
        }
        if result.is_err() {
            // A root can close between the two admission observations. Preserve
            // the original submit error while arranging shutdown on that race.
            let _ = self.observe_root();
        }
        result
    }

    /// The original completion remains available even after root revocation;
    /// the caller must preserve it and must not treat it as a valid label.
    pub fn take_completion(&mut self) -> Option<TeacherCompletion> {
        if self.terminal_ledger.unmatched.is_some() {
            return None;
        }
        let completion = self.owner
            .as_ref()
            .expect("retained teacher owner")
            .worker
            .take_completion()?;
        if self
            .terminal_ledger
            .pending
            .as_ref()
            .is_some_and(|dispatch| completion.matches_dispatch_identity(dispatch))
        {
            self.terminal_ledger.pending = None;
            // submitted was checked before issuing and only one dispatch is
            // pending, so delivered cannot overflow here.
            self.terminal_ledger.delivered += 1;
            Some(completion)
        } else {
            self.terminal_ledger.unmatched = Some(completion);
            None
        }
    }

    /// Inspect the original unmatched terminal without granting label or
    /// publication authority. It remains retained with the owner on drop.
    pub fn unmatched_completion(&self) -> Option<&TeacherCompletion> {
        self.terminal_ledger.unmatched.as_ref()
    }

    /// Preserve the original live-root and private-dispatch validation for a
    /// completion. Retirement evidence alone never makes a label eligible.
    /// Validate a completion against its original dispatch and live root.
    /// This does not itself authorize a training label or publication.
    pub fn validate_completion(
        &mut self,
        completion: &TeacherCompletion,
        dispatch: &TeacherDispatch,
    ) -> Result<(), CheckerError> {
        if !self
            .observe_root()
            .map_err(|_| invalid("teacher /2 root observation unavailable"))?
        {
            return Err(invalid("teacher /2 admission closed"));
        }
        let result = self
            .owner
            .as_ref()
            .expect("retained teacher owner")
            .validate_completion(completion, dispatch);
        if result.is_err() {
            let _ = self.observe_root();
        }
        result
    }

    /// Recover the lower worker's original diagnostic bytes only after join.
    pub fn take_diagnostics_after_join(&mut self) -> Option<TeacherDiagnostics> {
        if self.diagnostics != RoventTeacherDiagnosticsDispositionV2::NotObserved {
            return None;
        }
        let worker = &mut self.owner.as_mut().expect("retained teacher owner").worker;
        // Diagnostics can remain useful even when the original worker join
        // failed. Reading this clone does not consume the retained join result;
        // poll() continues to report that error separately from physical exit.
        let _join_snapshot = worker.shutdown_snapshot()?;
        let diagnostics = worker.take_diagnostics_after_join();
        self.diagnostics = if diagnostics.is_some() {
            RoventTeacherDiagnosticsDispositionV2::Recovered
        } else {
            RoventTeacherDiagnosticsDispositionV2::UnavailableOrAlreadyTaken
        };
        diagnostics
    }

    /// Poll without discarding reliable completions or consuming the owner.
    /// An error can coexist with separately confirmed process-exit evidence.
    pub fn poll(&mut self) -> RoventTeacherRetirementObservationV2 {
        if !self.shutdown_requested {
            let _ = self.observe_root();
        }
        let owner = self.owner.as_mut().expect("retained teacher owner");
        let joined = owner
            .worker
            .try_join()
            .or_else(|| owner.worker.shutdown_snapshot());
        let process = owner.worker.process_after_join();
        let occupancy = owner.worker.occupancy();
        let physical = if !self.shutdown_requested && joined.is_none() {
            RoventTeacherPhysicalRetirementV2::NotRequested
        } else if joined.is_none() {
            RoventTeacherPhysicalRetirementV2::Pending
        } else if let Some(evidence) = process.filter(|p| {
            p.process_identity.is_some()
                && p.exit_observed
                && p.stdout_drained
                && p.stderr_drained
                && p.cleanup_complete
                && !p.quarantined
                && !p.ownership_lost
        }) {
            RoventTeacherPhysicalRetirementV2::Confirmed(evidence)
        } else {
            RoventTeacherPhysicalRetirementV2::Unknown(process)
        };
        let terminal_accounting = match self.terminal_ledger.observation() {
            RoventTeacherTerminalAccountingV2::Pending { .. }
                if joined.is_some() && occupancy.reliable_queued == 0 =>
            {
                // A joined worker cannot later publish the missing terminal.
                RoventTeacherTerminalAccountingV2::Unknown
            }
            observation => observation,
        };
        RoventTeacherRetirementObservationV2 {
            root_observation_error: self.root_observation_error,
            worker_join: joined,
            physical,
            commands: occupancy.commands,
            reliable_queued: occupancy.reliable_queued,
            reliable_reserved: occupancy.reliable_reserved,
            diagnostics: self.diagnostics,
            terminal_accounting,
            registry_poisoned: self.registry_poisoned,
        }
    }

    /// Return the same owner only after physical closure, empty observable
    /// queues, observed diagnostics and matching delivery of every terminal.
    /// Delivery still does not prove label consumption or publication.
    /// Failure returns the wrapper itself so the caller can keep polling.
    pub fn try_finish(
        mut self,
    ) -> Result<
        (
            RoventCollectionTeacherOwnerV2,
            RoventTeacherRetirementObservationV2,
        ),
        Self,
    > {
        let mut observation = self.poll();
        if !matches!(
            &observation.physical,
            RoventTeacherPhysicalRetirementV2::Confirmed(_)
        ) || observation.worker_join.is_none()
            || observation.commands != 0
            || observation.reliable_queued != 0
            || observation.reliable_reserved != 0
            || observation.diagnostics == RoventTeacherDiagnosticsDispositionV2::NotObserved
            || !matches!(
                observation.terminal_accounting,
                RoventTeacherTerminalAccountingV2::Delivered { .. }
            )
        {
            return Err(self);
        }
        let poisoned = match self
            .reservation
            .as_mut()
            .and_then(|reservation| reservation.release().ok())
        {
            Some(poisoned) => poisoned,
            None => return Err(self),
        };
        observation.registry_poisoned |= poisoned;
        Ok((
            self.owner.take().expect("retained teacher owner"),
            observation,
        ))
    }

    /// Issue an opaque token only through the same physical retirement path.
    /// A failure returns the original lifecycle and retains recovery custody.
    pub fn try_retire(self) -> Result<RoventTeacherRetirementTokenV2, Self> {
        self.try_finish().map(|(owner, observation)| {
            RoventTeacherRetirementTokenV2 { owner, observation }
        })
    }
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
impl Drop for RoventCollectionTeacherLifecycleV2 {
    fn drop(&mut self) {
        if let Some(owner) = self.owner.take() {
            // No root mutex or worker operation is executed while holding the
            // registry lock. The reservation was made before this spawn.
            owner.worker.request_shutdown(self.cleanup_deadline);
            let parked = ParkedTeacherV2 {
                owner,
                shutdown_requested: true,
                root_observation_error: self.root_observation_error,
                diagnostics: self.diagnostics,
                cleanup_deadline: self.cleanup_deadline,
                registry_poisoned: self.registry_poisoned,
                terminal_ledger: std::mem::take(&mut self.terminal_ledger),
            };
            match self.reservation.as_mut() {
                Some(reservation) => {
                    if let Err(parked) = reservation.park(parked) {
                        // An impossible slot/custody violation must not drop
                        // an unconfirmed worker or pretend recovery succeeded.
                        quarantine_v2::mark_custody_violation();
                        std::mem::forget(parked);
                    }
                }
                None => {
                    quarantine_v2::mark_custody_violation();
                    std::mem::forget(parked);
                }
            }
        }
    }
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
impl Drop for RoventCollectionTeacherOwnerV2 {
    fn drop(&mut self) {
        // The lower worker requests shutdown on drop. Until its process and
        // pipes are demonstrably closed, quarantine the original source and
        // admission rather than releasing their lifetime with an unknown owner.
        let closed = self.worker.process_after_join().is_some_and(|p| {
            p.exit_observed
                && p.stdout_drained
                && p.stderr_drained
                && p.cleanup_complete
                && !p.quarantined
                && !p.ownership_lost
        });
        if !closed {
            if let Some(source) = self.bound_source.take() {
                std::mem::forget(source);
            }
            if let Some(custody) = self.custody.take() {
                std::mem::forget(custody);
            }
            if let Some(admission) = self.admission.take() {
                std::mem::forget(admission);
            }
        }
    }
}
