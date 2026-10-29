//! Pinned Rovent manifest endpoint → independent external UCI process owner.
//!
//! Profile parsing is declaration admission. Actual executable hashing, UCI
//! identification, option advertisement/ranges and ready are observed only by
//! the private worker. Neither stage proves a loaded NNUE's rights/precision,
//! native node credit, a Rules result or a complete Rovent execution receipt.

use crate::pals_checker_profile::rovent::{LoadedRoventExternalProfile, RoventExternalProfile};
use rz_contracts::Digest;
use rz_contracts::pals::SearchAuthority;
use rz_contracts::pals_adaptive::CpuWorkerId;
use rz_contracts::rovent::ExecutionMode;
use rz_experiments::{
    ArtifactRef, ExternalSourceV2, ExternalUciEndpointV2, NativeEngineRole,
    ROVENT_EXTERNAL_PROFILE_DOMAIN, RoventEngineDeclaration, RoventExternalProfilePin,
    RoventExternalPurpose, RoventInputLock, RoventInputLockV2, RoventResearcher,
};
use rz_search::cpu_checker::{
    CheckerError, CheckerShutdown, ExternalProcessIdentity, ExternalUciIdentity,
};
#[cfg(target_os = "linux")]
use rz_search::external_cpu::ExternalLaunchPin;
use rz_search::external_cpu::worker::ExternalWorkerClock;
use rz_search::external_cpu::{
    ExternalEnginePurpose, ExternalGoConstraint, ProfiledExternalCompletion,
    ProfiledExternalDiagnostics, ProfiledExternalDispatch, ProfiledExternalJob,
    ProfiledExternalRootFence, ProfiledExternalWorker,
};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

mod actual_root;
mod frontier_source;
pub use frontier_source::{RoventAdmittedExternalRootSource, RoventAdmittedExternalRootSourceV2};
mod collection_teacher;
pub use collection_teacher::RoventCollectionTeacherOwner;
#[cfg(feature = "onnx-cpu")]
pub use collection_teacher::RoventCollectionTeacherOwnerV2;
#[cfg(feature = "onnx-cpu")]
pub use collection_teacher::{
    BoundCollectionTeacherActorV2, BoundTeacherSpawnCauseV2, BoundTeacherSpawnFailureV2,
    RoventCollectionTeacherLifecycleV2, RoventTeacherDiagnosticsDispositionV2,
    RoventTeacherPolicyEvidenceV2, RoventTeacherPolicySealFailureV2, RoventTeacherPolicySealFailureKindV2,
    RoventTeacherPolicyCustodyV2, RoventTeacherPolicySealErrorV2,
    RoventTeacherPhysicalRetirementV2, RoventTeacherRetirementObservationV2,
    RoventTeacherRetirementTokenV2, RoventTeacherWorkerTerminalTokenV2,
    RoventTeacherNoChildRetirementTokenV2,
    RoventTeacherScopeAggregateAckV2, RoventTeacherScopeAggregateFailureV2,
    RoventTeacherRetiredActorV2,
    RoventTeacherTerminalAccountingV2, TeacherActorBindingErrorV2, TeacherRecoveryCustodianV2,
    TeacherRecoveryKeyV2, bind_collection_teacher_actor_v2, spawn_bound_collection_teacher_v2,
    verify_collection_teacher_pin,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum RoventExternalSlot {
    Researcher,
    Teacher { index: usize },
}

#[derive(Debug)]
pub struct PreparedRoventExternalLaunch {
    profile: LoadedRoventExternalProfile,
    binary: ArtifactRef,
    canonical_sha256: String,
    capability_sha256: String,
    input_lock_sha256: Option<String>,
    original_lock_v2: Option<Arc<RoventInputLockV2>>,
    mode: Option<ExecutionMode>,
    root_budget: Option<Digest>,
}

/// spawn 이전 실패의 원 source를 보존한다. low worker는 thread 생성 이후에는
/// 항상 owner를 반환하므로 실제 startup 실패는 이 source-only 오류가 아니라
/// 반환된 worker의 startup/physical cleanup 경로에 남는다.
#[derive(Debug)]
pub struct RoventExternalSpawnRejection {
    pub failure: CheckerError,
    pub launch: PreparedRoventExternalLaunch,
    pub executable: File,
    pub working_directory: File,
}

/// Only the retained worker's actual startup can construct this observation.
#[derive(Clone, Debug)]
pub struct VerifiedRoventExternalStartup {
    canonical_sha256: String,
    capability_sha256: String,
    input_lock_sha256: String,
    observed_uci: ExternalUciIdentity,
    process: ExternalProcessIdentity,
}

impl VerifiedRoventExternalStartup {
    pub fn canonical_sha256(&self) -> &str {
        &self.canonical_sha256
    }
    pub fn capability_sha256(&self) -> &str {
        &self.capability_sha256
    }
    pub fn input_lock_sha256(&self) -> &str {
        &self.input_lock_sha256
    }
    pub fn observed_uci(&self) -> &ExternalUciIdentity {
        &self.observed_uci
    }
    pub fn process_identity(&self) -> ExternalProcessIdentity {
        self.process
    }
    /// UCI ready does not independently read back option application.
    pub fn options_application_observed(&self) -> Option<bool> {
        None
    }
    pub fn model_loading_observed(&self) -> Option<bool> {
        None
    }
}

pub struct RoventExternalOwner {
    launch: PreparedRoventExternalLaunch,
    worker: ProfiledExternalWorker,
    initial_root: SearchAuthority,
    legacy_dispatched: bool,
    active_root: Option<actual_root::ActualExternalRoot>,
}

impl PreparedRoventExternalLaunch {
    /// The lock selects the endpoint; a caller cannot pass a teacher pin as R.
    /// This function opens no executable or model and never starts a process.
    pub fn from_lock(
        lock: &RoventInputLock,
        slot: RoventExternalSlot,
        profile_bytes: &[u8],
    ) -> Result<Self, CheckerError> {
        lock.validate()
            .map_err(|_| invalid("invalid Rovent input lock"))?;
        Self::from_validated_fields(
            &lock.manifest.engine,
            &lock.canonical_sha256,
            &lock.manifest.coordinator.original_root_budget_sha256,
            slot,
            profile_bytes,
        )
    }

    /// Select the slot from the actual /2 lock without creating a /1 lock.
    /// This still grants no process, root or teacher-result authority.
    pub fn from_lock_v2(
        lock: &Arc<RoventInputLockV2>,
        slot: RoventExternalSlot,
        profile_bytes: &[u8],
    ) -> Result<Self, CheckerError> {
        lock.validate()
            .map_err(|_| invalid("invalid Rovent input lock"))?;
        let mut launch = Self::from_validated_fields(
            &lock.manifest.engine,
            &lock.canonical_sha256,
            &lock.manifest.coordinator.original_root_budget_sha256,
            slot,
            profile_bytes,
        )?;
        launch.original_lock_v2 = Some(Arc::clone(lock));
        Ok(launch)
    }

    fn from_validated_fields(
        engine: &RoventEngineDeclaration,
        lock_sha256: &str,
        root_budget_sha256: &str,
        slot: RoventExternalSlot,
        profile_bytes: &[u8],
    ) -> Result<Self, CheckerError> {
        let pin = match slot {
            RoventExternalSlot::Researcher => match &engine.researcher {
                Some(RoventResearcher::External { profile }) => profile.as_ref(),
                _ => {
                    return Err(CheckerError::Unsupported(
                        "manifest has no external researcher",
                    ));
                }
            },
            RoventExternalSlot::Teacher { index } => engine.teachers.get(index).ok_or(
                CheckerError::Unsupported("manifest has no requested external teacher"),
            )?,
        };
        let expected = match slot {
            RoventExternalSlot::Researcher => RoventExternalPurpose::ResearchChecker,
            RoventExternalSlot::Teacher { .. } => RoventExternalPurpose::Teacher,
        };
        if pin.purpose != expected {
            return Err(invalid("manifest external slot/purpose mismatch"));
        }
        let mut launch = Self::from_pin(pin, profile_bytes)?;
        launch.input_lock_sha256 = Some(lock_sha256.to_owned());
        launch.mode = Some(engine.mode);
        launch.root_budget = Some(digest(root_budget_sha256)?);
        Ok(launch)
    }

    /// An arena profile uses the arena launch path, not a research worker.
    pub fn for_arena_profile(
        pin: &RoventExternalProfilePin,
        profile_bytes: &[u8],
    ) -> Result<Self, CheckerError> {
        if pin.purpose != RoventExternalPurpose::ArenaOpponent {
            return Err(invalid("arena launch requires an arena-opponent pin"));
        }
        Self::from_pin(pin, profile_bytes)
    }

    fn from_pin(pin: &RoventExternalProfilePin, bytes: &[u8]) -> Result<Self, CheckerError> {
        pin.profile
            .validate()
            .map_err(|_| invalid("invalid external profile artifact"))?;
        pin.binary
            .validate()
            .map_err(|_| invalid("invalid external binary artifact"))?;
        let profile = RoventExternalProfile::from_verified_bytes(bytes, &pin.profile.sha256)?;
        validate_pin(pin, &profile, bytes.len() as u64)?;
        Ok(Self {
            canonical_sha256: profile.registration().canonical_sha256.into(),
            capability_sha256: profile.profile().capability_sha256()?,
            profile,
            binary: pin.binary.clone(),
            input_lock_sha256: None,
            original_lock_v2: None,
            mode: None,
            root_budget: None,
        })
    }

    pub fn profile(&self) -> &RoventExternalProfile {
        self.profile.profile()
    }
    /// Borrow the original /2 allocation for an actual owner join. This does
    /// not establish a worker, root admission or physical-completion witness.
    pub fn original_lock_owner_v2(&self) -> Option<&Arc<RoventInputLockV2>> {
        self.original_lock_v2.as_ref()
    }
    pub fn canonical_sha256(&self) -> &str {
        &self.canonical_sha256
    }
    pub fn capability_sha256(&self) -> &str {
        &self.capability_sha256
    }
    pub fn binary(&self) -> &ArtifactRef {
        &self.binary
    }
    /// 선언된 설정만 반환한다. 원 FD 검증·startup·actual root·target의 증거가 아니다.
    pub fn configuration(
        &self,
    ) -> Result<rz_search::external_cpu::ExternalCpuConfig, CheckerError> {
        self.profile.configuration()
    }
    /// 기존 unstarted checker를 metadata 대조에 재사용한다. 이 호출은
    /// 프로세스/pipe를 시작하지 않으며 실제 worker는 별도 원 FD를 소비한다.
    pub fn checker_declaration(
        &self,
    ) -> Result<rz_search::external_cpu::ExternalUciCpuChecker, CheckerError> {
        self.profile.checker()
    }

    pub fn require_query(
        &self,
        constraint: ExternalGoConstraint,
        root_restricted: bool,
    ) -> Result<(), CheckerError> {
        self.profile
            .profile()
            .require_query(constraint, root_restricted)
    }

    /// A held native executable and cwd capability reach the original owner.
    /// The owner repeats full byte hashing before spawning; metadata admission
    /// here is deliberately not an actual executable-verification receipt.
    #[allow(clippy::too_many_arguments)]
    pub fn spawn_pinned(
        self,
        executable: File,
        working_directory: File,
        owner: CpuWorkerId,
        clock: ExternalWorkerClock,
        initial_root: SearchAuthority,
        startup_deadline: Instant,
        cancel: &AtomicBool,
    ) -> Result<RoventExternalOwner, CheckerError> {
        self.spawn_pinned_retaining(
            executable,
            working_directory,
            owner,
            clock,
            initial_root,
            startup_deadline,
            cancel,
        )
        .map_err(|rejection| rejection.failure)
    }

    /// 실제 source state는 실패 시 이 원 launch/FD를 다음 root 또는 명시
    /// 정리까지 보존한다. low worker에는 FD 복제만 넘기며 가중치 bytes는 복사하지 않는다.
    // Return the same launch and original FDs on pre-spawn rejection without
    // introducing an allocation that could itself lose that failure custody.
    #[allow(clippy::too_many_arguments, clippy::result_large_err)]
    pub fn spawn_pinned_retaining(
        self,
        executable: File,
        working_directory: File,
        owner: CpuWorkerId,
        clock: ExternalWorkerClock,
        initial_root: SearchAuthority,
        startup_deadline: Instant,
        cancel: &AtomicBool,
    ) -> Result<RoventExternalOwner, RoventExternalSpawnRejection> {
        let result = self.spawn_worker_from_original_descriptors(
            &executable,
            &working_directory,
            owner,
            clock,
            initial_root,
            startup_deadline,
            cancel,
        );
        match result {
            Ok(worker) => Ok(RoventExternalOwner {
                launch: self,
                worker,
                initial_root,
                legacy_dispatched: false,
                active_root: None,
            }),
            Err(failure) => Err(RoventExternalSpawnRejection {
                failure,
                launch: self,
                executable,
                working_directory,
            }),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn spawn_worker_from_original_descriptors(
        &self,
        executable: &File,
        working_directory: &File,
        owner: CpuWorkerId,
        clock: ExternalWorkerClock,
        initial_root: SearchAuthority,
        startup_deadline: Instant,
        cancel: &AtomicBool,
    ) -> Result<ProfiledExternalWorker, CheckerError> {
        if cancel.load(Ordering::Acquire) || Instant::now() >= startup_deadline {
            return Err(invalid("external startup canceled or expired"));
        }
        if self.input_lock_sha256.is_none() || self.mode.is_none() || self.root_budget.is_none() {
            return Err(CheckerError::Unsupported(
                "arena profile cannot start a research worker",
            ));
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (executable, working_directory, owner, clock, initial_root);
            Err(CheckerError::Unsupported(
                "Rovent external launch requires Linux native ELF",
            ))
        }
        #[cfg(target_os = "linux")]
        {
            let metadata = executable
                .metadata()
                .map_err(|_| invalid("external held executable metadata unavailable"))?;
            if !metadata.is_file()
                || metadata.len() != self.binary.bytes
                || metadata.len() == 0
                || metadata.len() > 256 * 1024 * 1024
            {
                return Err(invalid(
                    "external held executable size differs from manifest",
                ));
            }
            let executable = executable
                .try_clone()
                .map_err(|_| invalid("external original executable descriptor clone failed"))?;
            let working_directory = working_directory
                .try_clone()
                .map_err(|_| invalid("external original cwd descriptor clone failed"))?;
            let launch_pin =
                ExternalLaunchPin::from_owned_descriptors(executable, working_directory)?;
            let p = self.profile.profile();
            ProfiledExternalWorker::spawn(
                self.profile.configuration()?,
                p.engine,
                p.purpose,
                owner,
                clock,
                initial_root,
                startup_deadline,
                Some(launch_pin),
            )
        }
    }

    /// Existing V2 arena consumes a plain external-UCI endpoint: no RoveZero
    /// ONNX/CUDA/attestation arguments, model assets or environment overrides.
    /// Actual Fastchess startup, option advertisement and process evidence are
    /// still owned by arena and are not claimed by constructing this endpoint.
    pub fn arena_endpoint(
        &self,
        id: String,
        role: NativeEngineRole,
    ) -> Result<ExternalUciEndpointV2, CheckerError> {
        let p = self.profile.profile();
        if p.purpose != ExternalEnginePurpose::ArenaOpponent || self.input_lock_sha256.is_some() {
            return Err(CheckerError::Unsupported(
                "teacher/research profile cannot authorize an arena opponent",
            ));
        }
        if id.is_empty() || id.len() > 256 || id.chars().any(char::is_control) {
            return Err(invalid("arena endpoint id must be bounded text"));
        }
        Ok(ExternalUciEndpointV2 {
            id,
            role,
            family: match p.engine {
                rz_search::external_cpu::ExternalEngineProfile::Stockfish19EmbeddedNnue => {
                    "Stockfish"
                }
                rz_search::external_cpu::ExternalEngineProfile::Reckless090EmbeddedNnue => {
                    "Reckless"
                }
            }
            .into(),
            version: p.identity.declared_version.clone(),
            expected_uci_name: p.expected_uci_name.clone(),
            binary: self.binary.clone(),
            source: Some(ExternalSourceV2 {
                url: p.engine.source_repository().into(),
                commit: p.engine.source_commit().into(),
                license: p.identity.declared_license.clone(),
            }),
            arguments: Vec::new(),
            assets: Vec::new(),
            requested_options: p.identity.options.clone(),
            environment: None,
            handshake_timeout_ms: p.handshake_timeout_ms,
        })
    }
}

impl RoventExternalOwner {
    pub fn launch(&self) -> &PreparedRoventExternalLaunch {
        &self.launch
    }
    pub fn worker(&self) -> &ProfiledExternalWorker {
        &self.worker
    }
    pub fn try_join(&mut self) -> Option<Result<CheckerShutdown, CheckerError>> {
        self.worker.try_join()
    }
    pub fn try_take_diagnostics_after_join(&mut self) -> Option<ProfiledExternalDiagnostics> {
        self.worker.take_diagnostics_after_join()
    }

    /// Historical startup observation, tied to this owner's immutable lock and
    /// profile. It is not a current liveness/admission token. Assets absent or
    /// startup pending remain unknown; the final join is observed separately.
    pub fn verified_startup(&self) -> Result<Option<VerifiedRoventExternalStartup>, CheckerError> {
        let Some(startup) = self.worker.startup_report() else {
            return Ok(None);
        };
        let observed_uci = startup.result?;
        if observed_uci.name != self.launch.profile().expected_uci_name {
            return Err(invalid("actual external UCI name differs from profile"));
        }
        let attempt = startup
            .attempt
            .ok_or(invalid("actual external startup attempt missing"))?;
        let external = attempt
            .external
            .ok_or(invalid("actual external startup process missing"))?;
        let process = external.process;
        if external.request_id != 0 || external.partial_report.is_some() || !reusable(process) {
            return Err(invalid(
                "actual external startup origin evidence is incomplete",
            ));
        }
        Ok(Some(VerifiedRoventExternalStartup {
            canonical_sha256: self.launch.canonical_sha256.clone(),
            capability_sha256: self.launch.capability_sha256.clone(),
            input_lock_sha256: self
                .launch
                .input_lock_sha256
                .clone()
                .ok_or(invalid("external lock binding missing"))?,
            observed_uci,
            process: process
                .process_identity
                .ok_or(invalid("external process identity missing"))?,
        }))
    }

    pub fn submit(
        &mut self,
        job: ProfiledExternalJob,
    ) -> Result<ProfiledExternalDispatch, CheckerError> {
        require_v1_legacy_external_access(self.launch.original_lock_v2.is_some())?;
        if self.active_root.is_some() {
            return Err(CheckerError::Unsupported(
                "use actual-root external submission after live-root admission",
            ));
        }
        if Some(job.mode) != self.launch.mode
            || Some(job.question.root_budget) != self.launch.root_budget
        {
            return Err(invalid(
                "external job mode/root ledger differs from input lock",
            ));
        }
        self.launch
            .require_query(job.constraint, job.root_moves.is_some())?;
        let dispatch = self.worker.submit(job)?;
        self.legacy_dispatched = true;
        Ok(dispatch)
    }

    pub fn validate_completion(
        &self,
        completion: &ProfiledExternalCompletion,
        dispatch: &ProfiledExternalDispatch,
    ) -> Result<(), CheckerError> {
        require_v1_legacy_external_access(self.launch.original_lock_v2.is_some())?;
        if self.active_root.is_some() {
            return Err(CheckerError::Unsupported(
                "use actual-root completion validation after live-root admission",
            ));
        }
        let budget = self
            .launch
            .root_budget
            .ok_or(invalid("external exact root budget missing"))?;
        self.validate_completion_for_budget(completion, dispatch, budget)
    }

    fn validate_completion_for_budget(
        &self,
        completion: &ProfiledExternalCompletion,
        dispatch: &ProfiledExternalDispatch,
        root_budget: Digest,
    ) -> Result<(), CheckerError> {
        let actual_startup = self.verified_startup()?.ok_or(invalid(
            "actual external startup missing for completion validation",
        ))?;
        validate_owner_identity(
            Some(actual_startup.process_identity()),
            dispatch.process_identity(),
        )?;
        if !completion.matches_submission(dispatch)
            || completion.question().root_budget != root_budget
        {
            return Err(invalid(
                "external completion does not match original dispatch/ledger",
            ));
        }
        if let Ok(outcome) = completion.result() {
            let report = &outcome.report;
            let declared = self.launch.profile();
            let attempt = completion.attempt().ok_or(invalid(
                "external successful completion has no physical attempt",
            ))?;
            let external = attempt.external.as_ref().ok_or(invalid(
                "external successful completion has no process evidence",
            ))?;
            if report.profile != declared.engine
                || report.purpose != declared.purpose
                || report.raw.identity != declared.identity
                || report.raw.observed_uci.name != declared.expected_uci_name
                || report.requested != dispatch.constraint()
                || report.raw.requested_depth != report.requested.requested_depth().unwrap_or(0)
                || report.raw.root_restricted
                    != (dispatch.root_restriction()
                        != rz_search::external_cpu::worker::external_root_restriction_digest(None))
                || completion.go_issue() != rz_search::external_cpu::ExternalGoIssue::Issued
                || !report.readiness_barrier_completed
                || report.raw.request_id == 0
                || external.request_id != report.raw.request_id
                || external.partial_report.as_ref() != Some(&report.raw)
                || attempt.elapsed != report.raw.elapsed
                || external.process.process_identity != completion.process_identity()
                || !reusable(external.process)
            {
                return Err(invalid(
                    "external completion profile/constraint/physical provenance mismatch",
                ));
            }
        }
        // A valid failed/canceled terminal remains a failed/canceled terminal.
        // This check grants no successful teacher target or Rules authority.
        Ok(())
    }

    pub fn root_change_result(&self) -> Option<Result<ProfiledExternalRootFence, CheckerError>> {
        self.worker.root_change_result()
    }
}

fn require_v1_legacy_external_access(v2_lock: bool) -> Result<(), CheckerError> {
    if v2_lock {
        return Err(CheckerError::Unsupported(
            "V2 external dispatch and completion require actual-root admission",
        ));
    }
    Ok(())
}

fn validate_pin(
    pin: &RoventExternalProfilePin,
    loaded: &LoadedRoventExternalProfile,
    profile_bytes: u64,
) -> Result<(), CheckerError> {
    let p = loaded.profile();
    let expected_purpose = match pin.purpose {
        RoventExternalPurpose::Teacher => ExternalEnginePurpose::Teacher,
        RoventExternalPurpose::ResearchChecker => ExternalEnginePurpose::ResearchChecker,
        RoventExternalPurpose::ArenaOpponent => ExternalEnginePurpose::ArenaOpponent,
    };
    let threads = p
        .identity
        .options
        .get("Threads")
        .and_then(|value| value.parse::<u32>().ok());
    let hash = p
        .identity
        .options
        .get("Hash")
        .and_then(|value| value.parse::<u32>().ok());
    if pin.domain != ROVENT_EXTERNAL_PROFILE_DOMAIN
        || pin.profile.bytes != profile_bytes
        || pin.canonical_sha256 != loaded.registration().canonical_sha256
        || pin.capability_sha256 != p.capability_sha256()?
        || pin.purpose
            != match p.purpose {
                ExternalEnginePurpose::Teacher => RoventExternalPurpose::Teacher,
                ExternalEnginePurpose::ResearchChecker => RoventExternalPurpose::ResearchChecker,
                ExternalEnginePurpose::ArenaOpponent => RoventExternalPurpose::ArenaOpponent,
            }
        || p.purpose != expected_purpose
        || pin.binary.sha256 != p.identity.binary_sha256
        || pin.binary.license != p.identity.declared_license
        || pin.binary.bytes == 0
        || pin.binary.bytes > 256 * 1024 * 1024
        || pin.source_commit != p.engine.source_commit()
        || threads != Some(pin.requested_threads)
        || hash != Some(pin.requested_hash_mib)
    {
        return Err(invalid(
            "external manifest/profile/capability/binary/source/resource pins differ",
        ));
    }
    Ok(())
}

fn digest(value: &str) -> Result<Digest, CheckerError> {
    let mut bytes = [0; 32];
    if value.len() != 64 {
        return Err(invalid("external root budget digest length"));
    }
    for (at, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        let digit = |byte: u8| match byte {
            b'0'..=b'9' => Some(byte - b'0'),
            b'a'..=b'f' => Some(byte - b'a' + 10),
            _ => None,
        };
        bytes[at] = digit(pair[0])
            .zip(digit(pair[1]))
            .map(|(high, low)| high * 16 + low)
            .ok_or(invalid("external root budget digest encoding"))?;
    }
    Ok(Digest(bytes))
}
fn reusable(process: CheckerShutdown) -> bool {
    process
        .process_identity
        .is_some_and(|id| id.pid != 0 && id.process_group != 0 && id.proc_start_ticks != 0)
        && !process.quit_sent
        && !process.exit_observed
        && !process.cleanup_complete
        && !process.quarantined
        && !process.ownership_lost
}
fn validate_owner_identity(
    actual: Option<ExternalProcessIdentity>,
    submitted: ExternalProcessIdentity,
) -> Result<(), CheckerError> {
    if actual != Some(submitted) {
        return Err(invalid(
            "external completion belongs to another physical worker owner",
        ));
    }
    Ok(())
}
fn invalid(detail: &'static str) -> CheckerError {
    CheckerError::Invalid(detail)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pals_checker_profile::rovent::ROVENT_EXTERNAL_PROFILE_SCHEMA;
    use rz_search::cpu_checker::{
        ExternalCheckerIdentity, ExternalModelMetadata, ExternalTrainingKnowledge,
    };
    use rz_search::external_cpu::{
        EXTERNAL_UCI_SCORE_SEMANTICS, ExternalEngineProfile, ExternalEngineResources,
        arguments_sha256,
    };
    use sha2::{Digest as _, Sha256};

    fn fixture(purpose: ExternalEnginePurpose) -> (RoventExternalProfilePin, Vec<u8>) {
        let engine = ExternalEngineProfile::Reckless090EmbeddedNnue;
        let profile = RoventExternalProfile {
            schema: ROVENT_EXTERNAL_PROFILE_SCHEMA.into(),
            engine,
            purpose,
            program: "/opt/external/reckless".into(),
            working_directory: "/opt/external".into(),
            arguments: Vec::new(),
            identity: ExternalCheckerIdentity {
                adapter_semantics: EXTERNAL_UCI_SCORE_SEMANTICS.into(),
                binary_sha256: "a".repeat(64),
                launch_arguments_sha256: arguments_sha256(&[]),
                declared_name: engine.expected_uci_name().into(),
                declared_version: "0.9.0".into(),
                declared_source: format!(
                    "{}/tree/{}",
                    engine.source_repository(),
                    engine.source_commit()
                ),
                declared_license: engine.license().into(),
                options: engine
                    .options(purpose, ExternalEngineResources::default())
                    .unwrap(),
                assets: Vec::new(),
                model_metadata: ExternalModelMetadata {
                    weights_sha256: None,
                    training: ExternalTrainingKnowledge::Unknown,
                    declared_rights: None,
                    precision: None,
                },
            },
            expected_uci_name: engine.expected_uci_name().into(),
            max_depth: 64,
            max_prefix_plies: 4096,
            handshake_timeout_ms: 1000,
            max_task_wall_time_ms: 2000,
            stop_grace_ms: 500,
            shutdown_grace_ms: 500,
            max_output_bytes: 65536,
            max_line_bytes: 4096,
        };
        let bytes = serde_json::to_vec(&profile).unwrap();
        let hash = format!("{:x}", Sha256::digest(&bytes));
        let loaded = RoventExternalProfile::from_verified_bytes(&bytes, &hash).unwrap();
        let pin = RoventExternalProfilePin {
            domain: ROVENT_EXTERNAL_PROFILE_DOMAIN.into(),
            profile: ArtifactRef {
                path: "external/reckless-profile.json".into(),
                sha256: hash,
                bytes: bytes.len() as u64,
                source: "https://example.org/fixtures/rovent-external-profile".into(),
                license: "MIT".into(),
            },
            canonical_sha256: loaded.registration().canonical_sha256.into(),
            capability_sha256: profile.capability_sha256().unwrap(),
            purpose: match purpose {
                ExternalEnginePurpose::Teacher => RoventExternalPurpose::Teacher,
                ExternalEnginePurpose::ResearchChecker => RoventExternalPurpose::ResearchChecker,
                ExternalEnginePurpose::ArenaOpponent => RoventExternalPurpose::ArenaOpponent,
            },
            binary: ArtifactRef {
                path: "external/reckless-linux-generic".into(),
                sha256: "a".repeat(64),
                bytes: 100,
                source: engine.source_repository().into(),
                license: engine.license().into(),
            },
            source_commit: engine.source_commit().into(),
            requested_threads: 1,
            requested_hash_mib: 256,
        };
        (pin, bytes)
    }

    #[test]
    fn manifest_profile_purpose_capability_and_resources_are_bound_before_startup() {
        let (pin, bytes) = fixture(ExternalEnginePurpose::ResearchChecker);
        pin.profile
            .validate()
            .expect("fixture profile artifact has valid logical path/hash/bytes/source/license");
        pin.binary
            .validate()
            .expect("fixture engine artifact has valid logical path/hash/bytes/source/license");
        assert!(PreparedRoventExternalLaunch::from_pin(&pin, &bytes).is_ok());
        for case in 0..6 {
            let mut changed = pin.clone();
            match case {
                0 => changed.purpose = RoventExternalPurpose::Teacher,
                1 => changed.requested_threads = 2,
                2 => changed.requested_hash_mib = 512,
                3 => changed.capability_sha256 = "b".repeat(64),
                4 => changed.source_commit = "b".repeat(40),
                _ => changed.profile.source = "authored fixture".into(),
            }
            assert!(PreparedRoventExternalLaunch::from_pin(&changed, &bytes).is_err());
        }
    }

    #[test]
    fn arena_uses_only_its_own_engine_options_and_no_model_or_provider_arguments() {
        let (pin, bytes) = fixture(ExternalEnginePurpose::ArenaOpponent);
        let launch = PreparedRoventExternalLaunch::for_arena_profile(&pin, &bytes).unwrap();
        let endpoint = launch
            .arena_endpoint("reckless-opponent".into(), NativeEngineRole::Candidate)
            .unwrap();
        assert_eq!(endpoint.expected_uci_name, "Reckless 0.9.0");
        assert!(endpoint.arguments.is_empty());
        assert!(endpoint.assets.is_empty());
        assert!(endpoint.environment.is_none());
        assert!(endpoint.requested_options.contains_key("MoveOverhead"));
        for name in ["Ponder", "NumaPolicy", "EvalFile", "Move Overhead"] {
            assert!(!endpoint.requested_options.contains_key(name));
        }
        assert!(
            launch
                .require_query(ExternalGoConstraint::Nodes { reported_nodes: 32 }, false)
                .is_err()
        );
        let (pin, bytes) = fixture(ExternalEnginePurpose::Teacher);
        assert!(PreparedRoventExternalLaunch::for_arena_profile(&pin, &bytes).is_err());
    }

    #[test]
    fn prepared_profile_is_not_actual_startup_or_available_assets() {
        let (pin, bytes) = fixture(ExternalEnginePurpose::ResearchChecker);
        let launch = PreparedRoventExternalLaunch::from_pin(&pin, &bytes).unwrap();
        assert!(launch.input_lock_sha256.is_none());
        assert!(!launch.profile.registration().process_start_performed);
        assert_eq!(launch.profile.registration().model_loading_observed, None);
        assert_eq!(
            launch.profile.registration().options_application_observed,
            None
        );
        assert!(
            launch
                .require_query(ExternalGoConstraint::Nodes { reported_nodes: 32 }, false)
                .is_ok()
        );
        assert!(
            launch
                .require_query(ExternalGoConstraint::Nodes { reported_nodes: 32 }, true)
                .is_err()
        );
    }

    #[test]
    fn legacy_external_access_rejects_v2_without_live_root() {
        assert!(require_v1_legacy_external_access(false).is_ok());
        assert!(matches!(
            require_v1_legacy_external_access(true),
            Err(CheckerError::Unsupported(_))
        ));
    }

    #[test]
    fn matching_old_packet_and_dispatch_do_not_authorize_another_physical_owner() {
        let old = ExternalProcessIdentity {
            pid: 100,
            process_group: 100,
            proc_start_ticks: 1000,
        };
        let replacement = ExternalProcessIdentity {
            pid: 100,
            process_group: 100,
            proc_start_ticks: 1001,
        };
        assert!(validate_owner_identity(Some(old), old).is_ok());
        assert!(validate_owner_identity(None, old).is_err());
        assert!(validate_owner_identity(Some(replacement), old).is_err());
    }
}
