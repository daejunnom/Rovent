//! 원 retirement sink 등록 파일과 명시한 managed directory의 시작 경계.
//!
//! 등록은 원 lock의 고유 provenance 파일에서만 읽는다. 실제 sink는 단일
//! 소유권으로 Product에 이동하고 원 등록 FD는 serve 종료까지 별도로 보존한다.
//! 기본 경로 생성, native 실행, 질문 또는 과거 실행 권한 발급은 하지 않는다.

use super::super::{RoventStartupControl, process_boundary::VerifiedOwnedRoventProcessBoundaryV2};
use cap_fs_ext::DirExt;
use cap_std::fs::Dir;
use rz_contracts::Digest;
use rz_experiments::{ArtifactRef, ManifestError, RoventInputLock, RoventInputLockV2};
use rz_search::pals::engine::adaptive_port::profiled_retirement_archive::{
    ProfiledRetirementSinkAdmissionError, ProfiledRetirementSinkRegistration,
    RegisteredProfiledRetirementSink,
};
use rz_search::pals::store::{ArchiveRootCapability, StoreError};
use sha2::{Digest as _, Sha256};
use std::{
    error::Error,
    fmt,
    fs::File,
    io::Read,
    path::{Component, Path},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

const REGISTRATION_MAX_BYTES: u64 = 64 << 10;
const ROOT_PATH_MAX_BYTES: usize = 4096;

/// 실제 sink와 원 등록 파일을 함께 보존하는 시작 시점의 일회성 owner.
pub struct LaunchRetirementEvidence {
    sink: RegisteredProfiledRetirementSink,
    registration: Arc<LaunchRetirementRegistrationCustody>,
}

/// sink가 Product로 이동한 뒤에도 부모 launch가 보존할 원 등록 FD.
pub struct LaunchRetirementRegistrationCustody {
    input_lock_sha256: String,
    artifact: ArtifactRef,
    registration: ProfiledRetirementSinkRegistration,
    _file: File,
    // The original V2 process boundary owns the exact lock, directory FD and
    // startup control. Keep it alongside the registration until retirement
    // evidence is released, including on physical-completion quarantine.
    original_process_v2: Option<Arc<VerifiedOwnedRoventProcessBoundaryV2>>,
}

/// 플랫폼 미지원과 실제 입력·I/O·sink admission 오류를 구분해 보존한다.
#[derive(Debug)]
pub(super) enum LaunchRetirementEvidenceError {
    UnsupportedDurablePlatform,
    Cancelled,
    Expired,
    Invalid(&'static str),
    Allocation,
    Input(ManifestError),
    RegistrationJson(serde_json::Error),
    Root(StoreError),
    Sink(ProfiledRetirementSinkAdmissionError),
    Io {
        stage: &'static str,
        source: std::io::Error,
    },
}

impl fmt::Display for LaunchRetirementEvidenceError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedDurablePlatform => write!(
                output,
                "UnsupportedCapability: durable retirement evidence requires Linux"
            ),
            Self::Cancelled => write!(output, "retirement evidence admission cancelled"),
            Self::Expired => write!(
                output,
                "retirement filesystem admission scan deadline expired"
            ),
            Self::Invalid(detail) => write!(output, "retirement evidence admission: {detail}"),
            Self::Allocation => write!(output, "retirement registration allocation failed"),
            Self::Input(error) => write!(output, "retirement original input: {error}"),
            Self::RegistrationJson(error) => {
                write!(output, "retirement registration JSON: {error}")
            }
            Self::Root(error) => write!(output, "retirement directory capability: {error}"),
            Self::Sink(error) => write!(output, "{error}"),
            Self::Io { stage, source } => write!(output, "retirement {stage}: {source}"),
        }
    }
}
impl Error for LaunchRetirementEvidenceError {}

impl LaunchRetirementEvidence {
    pub fn input_lock_digest(&self) -> Digest {
        self.registration.input_lock_digest()
    }

    /// 엔진이나 NN 시작 전에 원 등록과 실제 directory capability를 인수한다.
    /// `admission_scan_deadline`과 `cancel`은 부모의 원 startup scan 창을 그대로
    /// 사용한다. 이 기한은 장기 UCI 수명이나 각 root의 persist 창이 아니다.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn open(
        lock: &RoventInputLock,
        private_root: &Dir,
        registration_snapshot_path: &str,
        managed_root: &Path,
        protected_execution_root: &Path,
        admission_scan_deadline: Instant,
        cancel: &AtomicBool,
    ) -> Result<Self, LaunchRetirementEvidenceError> {
        if !cfg!(target_os = "linux") {
            return Err(LaunchRetirementEvidenceError::UnsupportedDurablePlatform);
        }
        check(admission_scan_deadline, cancel)?;
        lock.validate()
            .map_err(LaunchRetirementEvidenceError::Input)?;
        Self::open_original(
            &lock.canonical_sha256,
            &lock.manifest.provenance,
            private_root,
            registration_snapshot_path,
            managed_root,
            protected_execution_root,
            admission_scan_deadline,
            cancel,
            None,
        )
    }

    /// Admit from the original V2 process owner and the same startup window.
    /// No reconstructed lock, reopened snapshot root or renewed deadline is
    /// accepted as a replacement for this authority.
    pub(super) fn open_v2(
        process: Arc<VerifiedOwnedRoventProcessBoundaryV2>,
        registration_snapshot_path: &str,
        managed_root: &Path,
        control: &RoventStartupControl,
    ) -> Result<Self, LaunchRetirementEvidenceError> {
        if !cfg!(target_os = "linux") {
            return Err(LaunchRetirementEvidenceError::UnsupportedDurablePlatform);
        }
        check(control.until, control.cancellation.as_ref())?;
        if !process.matches_startup_control(control) {
            return Err(LaunchRetirementEvidenceError::Invalid(
                "V2 retirement startup control differs from original process",
            ));
        }
        let lock = process.input_lock();
        lock.validate()
            .map_err(LaunchRetirementEvidenceError::Input)?;
        Self::open_original(
            &lock.canonical_sha256,
            &lock.manifest.provenance,
            process.original_root(),
            registration_snapshot_path,
            managed_root,
            process.private_cwd(),
            control.until,
            control.cancellation.as_ref(),
            Some(Arc::clone(&process)),
        )
    }

    pub(crate) fn original_lock_owner_v2(&self) -> Option<&Arc<RoventInputLockV2>> {
        self.registration
            .original_process_v2
            .as_ref()
            .map(|owner| owner.input_lock())
    }

    pub(crate) fn original_process_v2(&self) -> Option<&Arc<VerifiedOwnedRoventProcessBoundaryV2>> {
        self.registration.original_process_v2.as_ref()
    }

    #[allow(clippy::too_many_arguments)]
    fn open_original(
        input_lock_sha256: &str,
        provenance: &[ArtifactRef],
        private_root: &Dir,
        registration_snapshot_path: &str,
        managed_root: &Path,
        protected_execution_root: &Path,
        admission_scan_deadline: Instant,
        cancel: &AtomicBool,
        original_process_v2: Option<Arc<VerifiedOwnedRoventProcessBoundaryV2>>,
    ) -> Result<Self, LaunchRetirementEvidenceError> {
        if !cfg!(target_os = "linux") {
            return Err(LaunchRetirementEvidenceError::UnsupportedDurablePlatform);
        }
        check(admission_scan_deadline, cancel)?;
        if !rz_experiments::pals_private_snapshot_path_v5(registration_snapshot_path)
            || !protected_execution_root.is_absolute()
        {
            return Err(LaunchRetirementEvidenceError::Invalid(
                "closed registration path and original absolute execution root are required",
            ));
        }
        validate_managed_root(managed_root)?;
        let mut matching = provenance
            .iter()
            .filter(|artifact| artifact.path == registration_snapshot_path);
        let artifact = matching
            .next()
            .ok_or(LaunchRetirementEvidenceError::Invalid(
                "registration is absent from original provenance",
            ))?;
        if matching.next().is_some() {
            return Err(LaunchRetirementEvidenceError::Invalid(
                "registration is not a unique original provenance artifact",
            ));
        }
        let file = artifact
            .open_verified_at(private_root, REGISTRATION_MAX_BYTES)
            .map_err(LaunchRetirementEvidenceError::Input)?;
        let raw = read_registration(&file, artifact, admission_scan_deadline, cancel)?;
        let registration: ProfiledRetirementSinkRegistration = serde_json::from_slice(&raw)
            .map_err(LaunchRetirementEvidenceError::RegistrationJson)?;
        registration.validate().map_err(sink_error)?;
        check(admission_scan_deadline, cancel)?;

        // 경로를 기본값으로 보충하거나 directory를 만들지 않는다. 각 이름을
        // no-follow로 연 실제 Dir와 원 명시 경로를 기존 capability에 함께 준다.
        let directory = open_managed_root(managed_root, admission_scan_deadline, cancel)?;
        let capability =
            ArchiveRootCapability::from_directory(managed_root.to_path_buf(), directory)
                .map_err(LaunchRetirementEvidenceError::Root)?;
        check(admission_scan_deadline, cancel)?;
        let source_registration = registration.clone();
        let sink = RegisteredProfiledRetirementSink::admit(
            registration,
            capability,
            protected_execution_root,
            admission_scan_deadline,
            cancel,
        )
        .map_err(sink_error)?;
        Ok(Self {
            sink,
            registration: Arc::new(LaunchRetirementRegistrationCustody {
                input_lock_sha256: input_lock_sha256.to_owned(),
                artifact: artifact.clone(),
                registration: source_registration,
                _file: file,
                original_process_v2,
            }),
        })
    }

    /// 실제 sink는 한 번만 이동하고, 원 등록 FD는 부모의 serve 수명에 남긴다.
    /// 새 descriptor, 경로 재열기 또는 추가 I/O는 수행하지 않는다.
    pub(crate) fn into_parts(
        self,
    ) -> (
        RegisteredProfiledRetirementSink,
        Arc<LaunchRetirementRegistrationCustody>,
    ) {
        (self.sink, self.registration)
    }
}

impl LaunchRetirementRegistrationCustody {
    fn input_lock_digest(&self) -> Digest {
        Digest(
            rz_eval::asset::parse_sha256(&self.input_lock_sha256)
                .expect("validated original retirement registration lock digest"),
        )
    }

    pub fn input_lock_sha256(&self) -> &str {
        &self.input_lock_sha256
    }

    pub fn registration_artifact(&self) -> &ArtifactRef {
        &self.artifact
    }

    pub fn registration(&self) -> &ProfiledRetirementSinkRegistration {
        &self.registration
    }
}

fn read_registration(
    file: &File,
    artifact: &ArtifactRef,
    admission_scan_deadline: Instant,
    cancel: &AtomicBool,
) -> Result<Vec<u8>, LaunchRetirementEvidenceError> {
    check(admission_scan_deadline, cancel)?;
    let before = file
        .metadata()
        .map_err(|source| io("registration metadata", source))?;
    if !before.is_file()
        || before.len() != artifact.bytes
        || artifact.bytes == 0
        || artifact.bytes > REGISTRATION_MAX_BYTES
    {
        return Err(LaunchRetirementEvidenceError::Invalid(
            "original registration descriptor differs from its bounded artifact",
        ));
    }
    let length =
        usize::try_from(artifact.bytes).map_err(|_| LaunchRetirementEvidenceError::Allocation)?;
    let mut raw = Vec::new();
    raw.try_reserve_exact(length)
        .map_err(|_| LaunchRetirementEvidenceError::Allocation)?;
    file.take(REGISTRATION_MAX_BYTES + 1)
        .read_to_end(&mut raw)
        .map_err(|source| io("registration read", source))?;
    let after = file
        .metadata()
        .map_err(|source| io("registration metadata", source))?;
    if raw.len() != length
        || after.len() != artifact.bytes
        || before.modified().ok() != after.modified().ok()
        || format!("{:x}", Sha256::digest(&raw)) != artifact.sha256
    {
        return Err(LaunchRetirementEvidenceError::Invalid(
            "original registration bytes changed during exact descriptor read",
        ));
    }
    check(admission_scan_deadline, cancel)?;
    Ok(raw)
}

fn validate_managed_root(root: &Path) -> Result<(), LaunchRetirementEvidenceError> {
    if !root.is_absolute()
        || root.as_os_str().as_encoded_bytes().len() > ROOT_PATH_MAX_BYTES
        || !root
            .components()
            .any(|part| matches!(part, Component::Normal(_)))
        || root
            .components()
            .any(|part| !matches!(part, Component::RootDir | Component::Normal(_)))
    {
        return Err(LaunchRetirementEvidenceError::Invalid(
            "explicit absolute managed root must contain bounded normal directory names",
        ));
    }
    Ok(())
}

fn open_managed_root(
    root: &Path,
    admission_scan_deadline: Instant,
    cancel: &AtomicBool,
) -> Result<Dir, LaunchRetirementEvidenceError> {
    validate_managed_root(root)?;
    let mut directory = Dir::open_ambient_dir("/", cap_std::ambient_authority())
        .map_err(|source| io("managed root anchor", source))?;
    for component in root.components() {
        check(admission_scan_deadline, cancel)?;
        match component {
            Component::RootDir => {}
            Component::Normal(name) => {
                directory = directory
                    .open_dir_nofollow(name)
                    .map_err(|source| io("managed root no-follow directory", source))?;
            }
            _ => {
                return Err(LaunchRetirementEvidenceError::Invalid(
                    "managed root contains traversal or unsupported prefix",
                ));
            }
        }
    }
    Ok(directory)
}

fn check(until: Instant, cancel: &AtomicBool) -> Result<(), LaunchRetirementEvidenceError> {
    if cancel.load(Ordering::Acquire) {
        return Err(LaunchRetirementEvidenceError::Cancelled);
    }
    if Instant::now() >= until {
        return Err(LaunchRetirementEvidenceError::Expired);
    }
    Ok(())
}

fn sink_error(error: ProfiledRetirementSinkAdmissionError) -> LaunchRetirementEvidenceError {
    match error {
        ProfiledRetirementSinkAdmissionError::UnsupportedDurablePlatform => {
            LaunchRetirementEvidenceError::UnsupportedDurablePlatform
        }
        other => LaunchRetirementEvidenceError::Sink(other),
    }
}

fn io(stage: &'static str, source: std::io::Error) -> LaunchRetirementEvidenceError {
    LaunchRetirementEvidenceError::Io { stage, source }
}
