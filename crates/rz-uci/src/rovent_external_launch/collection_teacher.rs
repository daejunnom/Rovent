//! 독립 Collection CPU_T의 원 pinned launch 소유자 초안이다.
//! P-C 검색에 R을 추가하거나 기존 profiled R guard를 낮추지 않는다.
use super::*;
use crate::rovent_root_admission::RoventRootAdmission;
use rz_runtime::contracts::ContractClock;
#[cfg(target_os = "linux")]
use rz_search::external_cpu::collection_teacher::TeacherWorkerRegistration;
use rz_search::external_cpu::collection_teacher::{
    CollectionTeacherWorker, TeacherCompletion, TeacherDiagnostics, TeacherDispatch, TeacherJob,
};
use rz_search::external_cpu::process_scope::CollectionProcessPermit;
use std::sync::Arc;

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
    ) -> Result<Self, RoventCollectionTeacherSpawnRejectionV2> {
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
                    .accepts()
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
                    .accepts()
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
            .accepts();
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
            if let Some(custody) = self.custody.take() {
                std::mem::forget(custody);
            }
            if let Some(admission) = self.admission.take() {
                std::mem::forget(admission);
            }
        }
    }
}
