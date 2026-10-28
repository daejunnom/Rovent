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
