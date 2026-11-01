//! Linux NVIDIA ELF 로딩의 작은 FFI 소유 경계.
//!
//! 원래 bootstrap 소유자가 권리·출처·manifest를 확인한 신뢰 가능한 라이브러리를
//! 전용 디렉터리에 복사하고 readonly file capability로 넘겨야 한다. 실행 admission은
//! 총괄이 확인한 exact 19개 profile의 이름·크기·SHA-256에 고정하며 caller가 임의로
//! 발급한 digest를 trust로 취급하지 않는다. hash 일치는 native code sandbox가 아니다.
//! 성공과 부분 실패 모두 file pin과 native handle을 프로세스 종료까지 보존한다.

#![deny(unsafe_op_in_unsafe_fn)]

use std::error::Error;
use std::fmt;
use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;

/// System-installed library observation without copying native assets.
#[cfg(target_os = "linux")]
pub mod system;

#[cfg(target_os = "linux")]
use std::sync::atomic::{AtomicU8, Ordering};

// A process may use only one native ownership regime, including after failure.
// 0: unclaimed, 1: historical bundled owner, 2: system-library owner.
#[cfg(target_os = "linux")]
static NATIVE_OWNERSHIP_MODE: AtomicU8 = AtomicU8::new(0);

#[cfg(target_os = "linux")]
fn claim_native_mode(mode: u8) -> Result<(), LoadError> {
    match NATIVE_OWNERSHIP_MODE.compare_exchange(0, mode, Ordering::AcqRel, Ordering::Acquire) {
        Ok(_) => Ok(()),
        Err(existing) if existing == mode => Ok(()),
        Err(_) => Err(LoadError::new(
            LoadErrorKind::DifferentBundle,
            "native ownership regime differs from the existing process owner",
        )),
    }
}

#[cfg(feature = "ort-bindings")]
pub mod ort_binding;

/// Retained-file OS ownership helpers, kept outside safe UCI consumers.
#[cfg(windows)]
pub mod pinned_reader;

/// 확인한 CUDA 12/cuDNN 9 profile의 dependency-first 로딩 순서. 부분 집합은 거부한다.
pub const NVIDIA_LOAD_ORDER: [&str; 16] = [
    "libcudart.so.12",
    "libnvJitLink.so.12",
    "libnvrtc-builtins.so.12.8",
    "libnvrtc.so.12",
    "libcublasLt.so.12",
    "libcublas.so.12",
    "libcurand.so.10",
    "libcufft.so.11",
    "libcudnn_graph.so.9",
    "libcudnn_ops.so.9",
    "libcudnn_adv.so.9",
    "libcudnn_cnn.so.9",
    "libcudnn_engines_precompiled.so.9",
    "libcudnn_engines_runtime_compiled.so.9",
    "libcudnn_heuristic.so.9",
    "libcudnn.so.9",
];

const EAGER_INDICES: [usize; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
const SHIM_LAZY_INDICES: [usize; 9] = [0, 1, 2, 3, 4, 5, 6, 7, 15];

/// Loading behavior is separate from the unchanged, complete 19-file trust
/// profile. The lazy profile is an explicit diagnostic candidate, not a new
/// default or evidence that preload caused a native termination failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeLoadingProfile {
    EagerCuda12Cudnn9V1,
    CuDnnShimLazyV1,
}

impl NativeLoadingProfile {
    pub const fn identifier(self) -> &'static str {
        match self {
            Self::EagerCuda12Cudnn9V1 => "eager-cuda12-cudnn9-v1",
            Self::CuDnnShimLazyV1 => "experimental-cudnn-shim-lazy-v1",
        }
    }

    /// Canonical policy bytes; callers may hash these separately from the
    /// unchanged binary bundle digest. Indices refer to NVIDIA_LOAD_ORDER.
    pub const fn canonical_descriptor(self) -> &'static str {
        match self {
            Self::EagerCuda12Cudnn9V1 => {
                "rz-native-loading-profile/1;eager-cuda12-cudnn9-v1;declared-nvidia=16;declared-ort=3;rtld=now-global;eager=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15;deferred=none;resident=exact-pins"
            }
            Self::CuDnnShimLazyV1 => {
                "rz-native-loading-profile/1;experimental-cudnn-shim-lazy-v1;declared-nvidia=16;declared-ort=3;rtld=now-global;eager=0,1,2,3,4,5,6,7,15;deferred=8,9,10,11,12,13,14;resident=exact-pins"
            }
        }
    }

    pub const fn eager_indices(self) -> &'static [usize] {
        match self {
            Self::EagerCuda12Cudnn9V1 => &EAGER_INDICES,
            Self::CuDnnShimLazyV1 => &SHIM_LAZY_INDICES,
        }
    }
}

/// A bounded origin observation at one audit boundary. An absent deferred
/// image is not claimed resident or unused. This is not kernel placement,
/// device completion, allocator residency, VRAM, or normal-process-exit proof.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeMappingObservation {
    pub profile: NativeLoadingProfile,
    pub declared_nvidia_files: usize,
    pub required_nvidia_files: Vec<&'static str>,
    pub mapped_nvidia_files: Vec<&'static str>,
    pub deferred_nvidia_not_mapped: Vec<&'static str>,
    /// None means the ORT images were outside this audit's scope.
    pub mapped_ort_files: Option<Vec<&'static str>>,
}

pub const MAX_LIBRARY_BYTES: u64 = 1024 * 1024 * 1024;
pub const MAX_TOTAL_BYTES: u64 = 4 * MAX_LIBRARY_BYTES;
pub const MAX_LIBRARY_COUNT: usize = 32;
pub const MAX_MAPS_BYTES: usize = 2 * 1024 * 1024;
#[cfg(any(target_os = "linux", test))]
const MAX_DIAGNOSTIC_BYTES: usize = 1024;

pub const ORT_LIBRARY_NAMES: [&str; 3] = [
    "libonnxruntime.so.1.22.0",
    "libonnxruntime_providers_shared.so",
    "libonnxruntime_providers_cuda.so",
];

/// 총괄이 공개 wheel metadata와 실제 추출 파일을 대조하여 고정한 실행 admission.
/// 다른 release/profile은 별도 source review와 실제 backend 인수를 거쳐 추가한다.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TrustedNativeProfileEntry {
    pub filename: &'static str,
    pub bytes: u64,
    pub digest: [u8; 32],
}

pub const TRUSTED_NATIVE_PROFILE: [TrustedNativeProfileEntry; 19] = [
    trusted(
        "libcudart.so.12",
        728800,
        "218eec4c8385a32e258a0235be4d449986844f2d0de3430052f0924e6fe60f71",
    ),
    trusted(
        "libnvJitLink.so.12",
        94101392,
        "0369e6867d44b800437de4e146d72c65afc6c75adf677a15c2ecd8e6a7ac135f",
    ),
    trusted(
        "libnvrtc-builtins.so.12.8",
        6338504,
        "eccaa824230ee7858a94a3055cb01f1cb634df05d313880d0bdcf195161fcb4e",
    ),
    trusted(
        "libnvrtc.so.12",
        104487248,
        "43731e24cd89e3749826304f304e8aa11fbecf1188715271b1f5018d6212b5e6",
    ),
    trusted(
        "libcublasLt.so.12",
        751771728,
        "10b5e6631cf8115c661eb895ed1533826308b58f7956466f53d236a40c9b622c",
    ),
    trusted(
        "libcublas.so.12",
        116388640,
        "031ce6c2cbfbb9468f040527cab5c599069ce5609e73e28f87503881063eac21",
    ),
    trusted(
        "libcurand.so.10",
        136749240,
        "f9bea038a2703b721571fd45a299a898141fd8cb264a5912635c95116f5960fe",
    ),
    trusted(
        "libcufft.so.11",
        278925016,
        "5c912146449614f9d73ebd1a5cb604242da6b819d94ba5b4a99272a0649f3761",
    ),
    trusted(
        "libcudnn_graph.so.9",
        4391696,
        "7b2abeb742ad5b737aeb50185481168b325ab443c87b66de3ceece6431e0e771",
    ),
    trusted(
        "libcudnn_ops.so.9",
        119547416,
        "e822b34d447d2c83f275d2e1e6373a88c19dcadd7a7d289ed27d8ec52a210323",
    ),
    trusted(
        "libcudnn_adv.so.9",
        255866448,
        "c7f2859225c2fc2992235d3a2504f641fa8ec83a1996a354b07609db8208a746",
    ),
    trusted(
        "libcudnn_cnn.so.9",
        6300904,
        "3d1643d323d9dba75236d5b9a7741a5b6606ea6ffbcf0a6268c0c6e6d4cb3dd4",
    ),
    trusted(
        "libcudnn_engines_precompiled.so.9",
        583023472,
        "4ab4bb62c6291d0a3650feb583e62693bdc6a5023352dbc64231187b2ec9cfdc",
    ),
    trusted(
        "libcudnn_engines_runtime_compiled.so.9",
        27276224,
        "20aab41f9c36e846ba4ccc6e59b67251becfbf992e87d281ffaadbf4f24f32c5",
    ),
    trusted(
        "libcudnn_heuristic.so.9",
        56627064,
        "6ad33b511a6f9b880c9c13018a669a6d818cddf6842700f5403906fa97eb5279",
    ),
    trusted(
        "libcudnn.so.9",
        125136,
        "72f74476dcdc074b9d56f8c5cf576ceff6a595989832b9b0798478dc48a4ffcd",
    ),
    trusted(
        "libonnxruntime.so.1.22.0",
        19950288,
        "09fb71acf9debf4c5755f2c49522bfa80e580223be83d3f4e18744788d4be7db",
    ),
    trusted(
        "libonnxruntime_providers_shared.so",
        14632,
        "ee7bf6d4d32f523dbdebf9dec70a1b3287a1a60d019192490732750dc385f98c",
    ),
    trusted(
        "libonnxruntime_providers_cuda.so",
        407530104,
        "6b9677938c96bf988a0a06958594778dc4b516a2d8ab8d4f54645195e29101d0",
    ),
];

/// Exact additional CUDA runtime declaration from NVIDIA's unmodified
/// nvidia_cuda_runtime_cu12-12.8.90 manylinux2014_x86_64 wheel.
/// Wheel SHA-256: adade8dcbd0edf427b7204d480d6066d33902cab2a4707dcfc48a2d0fd44ab90.
/// Member: nvidia/cuda_runtime/lib/libcudart.so.12. The remaining 18 entries,
/// file-content/FD checks and complete-bundle requirement stay unchanged.
/// Keep the historical declaration above readable for existing artifacts.
pub const TRUSTED_CUDART_12_8_90_WHEEL: TrustedNativeProfileEntry = trusted(
    "libcudart.so.12",
    728800,
    "c3a75b33af334a3486d197dbd1584a2985183ba4688d237a2be5f2f679329920",
);

const fn trusted(filename: &'static str, bytes: u64, sha256: &str) -> TrustedNativeProfileEntry {
    let encoded = sha256.as_bytes();
    assert!(
        encoded.len() == 64,
        "trusted digest must have 64 hex characters"
    );
    let mut digest = [0_u8; 32];
    let mut index = 0;
    while index < digest.len() {
        digest[index] = (hex_digit(encoded[index * 2]) << 4) | hex_digit(encoded[index * 2 + 1]);
        index += 1;
    }
    TrustedNativeProfileEntry {
        filename,
        bytes,
        digest,
    }
}

const fn hex_digit(byte: u8) -> u8 {
    match byte {
        b'0'..=b'9' => byte - b'0',
        b'a'..=b'f' => byte - b'a' + 10,
        _ => panic!("trusted digest must use lowercase hex"),
    }
}

/// ORT native 호출 전에 exact 3개의 이름·크기·digest 선언을 고정 profile과 대조한다.
/// 실제 file 내용/소유권 검증은 원래 bootstrap에서 먼저 수행한다. native 코드는 실행하지 않는다.
pub fn validate_runtime_profile(libraries: &[OwnedLibrary]) -> Result<(), LoadError> {
    #[cfg(target_os = "linux")]
    {
        linux::validate_runtime_profile(libraries)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = libraries;
        Err(LoadError::new(
            LoadErrorKind::UnsupportedPlatform,
            "runtime ELF profile is supported only on Linux",
        ))
    }
}

/// 원래 bootstrap 소유자로부터 이전받은 readonly file capability.
///
/// `path`는 exact absolute copied path, `digest`는 확인한 manifest의 SHA-256이다.
/// 이 타입이 새로운 trust를 발급하지 않는다. 외부 사용자의 임의 hash/path를 받는
/// plugin loader로 노출하지 않는다.
pub struct OwnedLibrary {
    pub path: PathBuf,
    pub file: File,
    pub digest: [u8; 32],
    pub bytes: u64,
}

impl fmt::Debug for OwnedLibrary {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OwnedLibrary")
            .field("bytes", &self.bytes)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LoadErrorKind {
    UnsupportedPlatform,
    InvalidBundle,
    UntrustedProfile,
    ResourceLimit,
    AmbientEnvironment,
    FileIdentity,
    FileAccess,
    InvalidMaps,
    /// System mapping observation could not be read or interpreted. This is
    /// evidence unavailable, not proof of a conflicting native mapping.
    ObservationUnavailable,
    ForeignMapping,
    MissingMapping,
    MappingIdentity,
    NativeLoad,
    DifferentBundle,
    ProcessPoisoned,
}

/// native 원문은 명시적 로컬 accessor로만 얻는다. 기본 출력에는 원문/경로가 없다.
#[derive(Clone)]
pub struct LoadError {
    pub kind: LoadErrorKind,
    pub detail: &'static str,
    diagnostic: Option<Arc<str>>,
    diagnostic_truncated: bool,
}

impl LoadError {
    fn new(kind: LoadErrorKind, detail: &'static str) -> Self {
        Self {
            kind,
            detail,
            diagnostic: None,
            diagnostic_truncated: false,
        }
    }

    #[cfg(any(target_os = "linux", test))]
    fn with_diagnostic(mut self, original: String) -> Self {
        let mut end = original.len().min(MAX_DIAGNOSTIC_BYTES);
        while !original.is_char_boundary(end) {
            end -= 1;
        }
        self.diagnostic_truncated = end < original.len();
        self.diagnostic = Some(Arc::from(&original[..end]));
        self
    }

    /// 원본의 최대 1024 UTF-8 bytes. private path가 포함될 수 있어 자동 출력하지 않는다.
    pub fn diagnostic(&self) -> Option<&str> {
        self.diagnostic.as_deref()
    }

    /// diagnostic이 원문 전체가 아닌 bounded prefix이면 true다.
    pub const fn diagnostic_truncated(&self) -> bool {
        self.diagnostic_truncated
    }

    pub const fn cause_code(&self) -> &'static str {
        match self.kind {
            LoadErrorKind::UnsupportedPlatform => "native.unsupported_platform",
            LoadErrorKind::InvalidBundle => "native.invalid_bundle",
            LoadErrorKind::UntrustedProfile => "native.untrusted_profile",
            LoadErrorKind::ResourceLimit => "native.resource_limit",
            LoadErrorKind::AmbientEnvironment => "native.ambient_environment",
            LoadErrorKind::FileIdentity => "native.file_identity",
            LoadErrorKind::FileAccess => "native.file_access",
            LoadErrorKind::InvalidMaps => "native.invalid_maps",
            LoadErrorKind::ObservationUnavailable => "native.observation_unavailable",
            LoadErrorKind::ForeignMapping => "native.foreign_mapping",
            LoadErrorKind::MissingMapping => "native.missing_mapping",
            LoadErrorKind::MappingIdentity => "native.mapping_identity",
            LoadErrorKind::NativeLoad => "native.dlopen",
            LoadErrorKind::DifferentBundle => "native.different_bundle",
            LoadErrorKind::ProcessPoisoned => "native.process_poisoned",
        }
    }
}

impl fmt::Display for LoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.cause_code(), self.detail)
    }
}

impl fmt::Debug for LoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LoadError")
            .field("kind", &self.kind)
            .field("detail", &self.detail)
            .field("has_diagnostic", &self.diagnostic.is_some())
            .field("diagnostic_truncated", &self.diagnostic_truncated)
            .finish()
    }
}

impl Error for LoadError {}

/// 프로세스 전역의 단일 NVIDIA bundle admission.
pub struct LibrarySet;

impl LibrarySet {
    /// 명시적인 초기 GPU bootstrap에서 한 번 호출한다. ambient loader 환경은 변경하지 않는다.
    pub fn load(libraries: Vec<OwnedLibrary>) -> Result<ProcessLibrarySet, LoadError> {
        Self::load_with_profile(libraries, NativeLoadingProfile::EagerCuda12Cudnn9V1)
    }

    /// Explicit experimental bootstrap. Every NVIDIA file must still be
    /// declared, hash-checked and pinned; the policy changes only direct load
    /// order and the mandatory mapped subset. Never selected after a failure.
    pub fn load_with_profile(
        libraries: Vec<OwnedLibrary>,
        profile: NativeLoadingProfile,
    ) -> Result<ProcessLibrarySet, LoadError> {
        #[cfg(target_os = "linux")]
        {
            linux::load(libraries, profile)
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = profile;
            drop(libraries);
            Err(LoadError::new(
                LoadErrorKind::UnsupportedPlatform,
                "NVIDIA ELF loading is supported only on Linux",
            ))
        }
    }
}

/// Drop으로 unload하지 않는 process-global pin의 복제 가능한 관찰 토큰.
#[derive(Clone)]
pub struct ProcessLibrarySet {
    #[cfg(target_os = "linux")]
    inner: Arc<linux::ProcessPin>,
}

impl ProcessLibrarySet {
    pub fn loading_profile(&self) -> NativeLoadingProfile {
        #[cfg(target_os = "linux")]
        {
            self.inner.profile
        }
        #[cfg(not(target_os = "linux"))]
        {
            // A token cannot be constructed on these platforms.
            NativeLoadingProfile::EagerCuda12Cudnn9V1
        }
    }

    /// ORT session 초기화 전후와 warm probe 전후에 호출한다.
    ///
    /// NVIDIA mapping만 검증한다. ORT core/providers는 원래 ORT bootstrap가 별도
    /// 검증한다. 최초 검증 실패는 영구 latch되어 이후 같은 토큰으로도 회복하지 않는다.
    pub fn verify_mappings(&self) -> Result<(), LoadError> {
        #[cfg(target_os = "linux")]
        {
            linux::verify(self)
        }
        #[cfg(not(target_os = "linux"))]
        {
            self.mapping_observation().map(|_| ())
        }
    }

    pub fn mapping_observation(&self) -> Result<NativeMappingObservation, LoadError> {
        #[cfg(target_os = "linux")]
        {
            linux::observe(self)
        }
        #[cfg(not(target_os = "linux"))]
        {
            Err(LoadError::new(
                LoadErrorKind::UnsupportedPlatform,
                "NVIDIA ELF mapping verification is supported only on Linux",
            ))
        }
    }

    /// ORT 초기화와 첫 provider/warm probe 이후 exact ORT 3개와 NVIDIA 전체 pin을 확인한다.
    /// eager는 16개 resident가 필수이며, 명시적 shim-lazy는 roots9가 필수다.
    /// resident인 모든 deferred image도 동일한 전체 pin과 대조한다.
    ///
    /// ORT 원래 소유자가 넘긴 readonly descriptor를 최초 호출에서 복제하여 process pin에
    /// 추가한다. 이후 다른 ORT bundle은 거부하며 모든 추가 검증 실패도 최초 실패로 latch된다.
    pub fn verify_runtime_mappings(&self, ort: &[OwnedLibrary]) -> Result<(), LoadError> {
        #[cfg(target_os = "linux")]
        {
            linux::verify_runtime(self, ort)
        }
        #[cfg(not(target_os = "linux"))]
        {
            self.runtime_mapping_observation(ort).map(|_| ())
        }
    }

    pub fn runtime_mapping_observation(
        &self,
        ort: &[OwnedLibrary],
    ) -> Result<NativeMappingObservation, LoadError> {
        #[cfg(target_os = "linux")]
        {
            linux::observe_runtime(self, ort)
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = ort;
            Err(LoadError::new(
                LoadErrorKind::UnsupportedPlatform,
                "runtime ELF mapping verification is supported only on Linux",
            ))
        }
    }
}

impl fmt::Debug for ProcessLibrarySet {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProcessLibrarySet")
            .field("nvidia_library_count", &NVIDIA_LOAD_ORDER.len())
            .field("loading_profile", &self.loading_profile())
            .finish_non_exhaustive()
    }
}

#[cfg(target_os = "linux")]
mod linux {
    use super::*;
    use libloading::os::unix::{Library, RTLD_GLOBAL, RTLD_NOW};
    use sha2::{Digest, Sha256};
    use std::collections::BTreeSet;
    use std::fs::{self, Metadata};
    use std::io::Read;
    use std::mem::ManuallyDrop;
    use std::os::fd::AsRawFd;
    use std::os::unix::fs::{FileExt, MetadataExt};
    use std::path::Path;
    use std::sync::{Mutex, OnceLock};

    const MAX_PATH_BYTES: usize = 4096;
    const MAX_PATH_DEPTH: usize = 128;
    const MAX_FDINFO_BYTES: usize = 4096;
    const AMBIENT_VARIABLES: [&str; 7] = [
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "LD_AUDIT",
        "LD_DEBUG",
        "LD_PROFILE",
        "LD_ORIGIN_PATH",
        "CUDA_INJECTION64_PATH",
    ];

    struct FilePin {
        path: PathBuf,
        file: ManuallyDrop<File>,
        digest: [u8; 32],
        bytes: u64,
    }

    impl From<OwnedLibrary> for FilePin {
        fn from(input: OwnedLibrary) -> Self {
            Self {
                path: input.path,
                file: ManuallyDrop::new(input.file),
                digest: input.digest,
                bytes: input.bytes,
            }
        }
    }

    #[derive(Clone, Debug)]
    struct ExpectedMapping {
        path: PathBuf,
        dev_major: u64,
        dev_minor: u64,
        inode: u64,
    }

    pub(super) struct ProcessPin {
        fingerprint: [u8; 32],
        pub(super) profile: NativeLoadingProfile,
        files: Vec<FilePin>,
        handles: Vec<ManuallyDrop<Library>>,
        expected: Vec<ExpectedMapping>,
    }

    #[derive(Default)]
    struct ProcessLatch {
        attempt: Option<Attempt>,
    }

    struct Attempt {
        fingerprint: [u8; 32],
        pin: Arc<ProcessPin>,
        outcome: Result<(), LoadError>,
        runtime: Option<RuntimePin>,
    }

    struct RuntimePin {
        fingerprint: [u8; 32],
        files: Vec<FilePin>,
        expected: Vec<ExpectedMapping>,
    }

    impl Attempt {
        fn check_profile(&mut self, profile: NativeLoadingProfile) -> Result<(), LoadError> {
            if self.pin.profile != profile {
                let error = LoadError::new(
                    LoadErrorKind::DifferentBundle,
                    "process already attempted another native loading policy",
                );
                return Err(self.retain_failure(error));
            }
            self.outcome.clone()
        }

        fn check_bundle(&self, fingerprint: [u8; 32]) -> Result<(), LoadError> {
            if self.fingerprint != fingerprint {
                return Err(LoadError::new(
                    LoadErrorKind::DifferentBundle,
                    "process already attempted another native bundle",
                ));
            }
            self.outcome.clone()
        }

        fn retain_failure(&mut self, error: LoadError) -> LoadError {
            if let Err(first) = &self.outcome {
                return first.clone();
            }
            self.outcome = Err(error.clone());
            error
        }
    }

    static PROCESS: OnceLock<Mutex<ProcessLatch>> = OnceLock::new();

    fn process() -> &'static Mutex<ProcessLatch> {
        PROCESS.get_or_init(|| Mutex::new(ProcessLatch::default()))
    }

    fn poisoned() -> LoadError {
        LoadError::new(
            LoadErrorKind::ProcessPoisoned,
            "process native admission is poisoned; restart the process",
        )
    }

    pub(super) fn load(
        mut libraries: Vec<OwnedLibrary>,
        profile: NativeLoadingProfile,
    ) -> Result<ProcessLibrarySet, LoadError> {
        let mut latch = process().lock().map_err(|_| poisoned())?;
        let sorted = sort_inputs(&mut libraries);
        let fingerprint = input_fingerprint(&libraries);
        if let Some(attempt) = &mut latch.attempt {
            attempt.check_bundle(fingerprint)?;
            attempt.check_profile(profile)?;
            // 같은 manifest라도 새 descriptor/path를 신뢰하지 않고 다시 확인한다.
            let mut supplied = ProcessPin::new(fingerprint, profile, libraries);
            let checked = sorted.and_then(|()| preflight(&mut supplied));
            close_temporary_files(supplied.files);
            if let Err(error) =
                checked.and_then(|()| audit(&attempt.pin, profile.eager_indices()).map(|_| ()))
            {
                return Err(attempt.retain_failure(error));
            }
            return Ok(ProcessLibrarySet {
                inner: Arc::clone(&attempt.pin),
            });
        }

        // File/handle는 preflight 및 native 실패·Rust unwind에도 Drop으로 해제하지 않는다.
        // static latch가 보존되며 lock poison 이후에도 프로세스 restart만 허용한다.
        let mut pin = ProcessPin::new(fingerprint, profile, libraries);
        let outcome = sorted.and_then(|()| initialize(&mut pin));
        let pin = Arc::new(pin);
        latch.attempt = Some(Attempt {
            fingerprint,
            pin: Arc::clone(&pin),
            outcome: outcome.clone(),
            runtime: None,
        });
        outcome?;
        Ok(ProcessLibrarySet { inner: pin })
    }

    impl ProcessPin {
        fn new(
            fingerprint: [u8; 32],
            profile: NativeLoadingProfile,
            libraries: Vec<OwnedLibrary>,
        ) -> Self {
            Self {
                fingerprint,
                profile,
                files: libraries
                    .into_iter()
                    .take(MAX_LIBRARY_COUNT)
                    .map(FilePin::from)
                    .collect(),
                handles: Vec::with_capacity(NVIDIA_LOAD_ORDER.len()),
                expected: Vec::with_capacity(NVIDIA_LOAD_ORDER.len()),
            }
        }
    }

    pub(super) fn verify(set: &ProcessLibrarySet) -> Result<(), LoadError> {
        verify_seen(set).map(|_| ())
    }

    pub(super) fn observe(set: &ProcessLibrarySet) -> Result<NativeMappingObservation, LoadError> {
        verify_seen(set).map(|seen| observation(set.inner.profile, &seen, false))
    }

    fn verify_seen(set: &ProcessLibrarySet) -> Result<BTreeSet<usize>, LoadError> {
        let mut latch = process().lock().map_err(|_| poisoned())?;
        let attempt = latch.attempt.as_mut().ok_or_else(poisoned)?;
        if attempt.fingerprint != set.inner.fingerprint || !Arc::ptr_eq(&attempt.pin, &set.inner) {
            return Err(LoadError::new(
                LoadErrorKind::DifferentBundle,
                "mapping token does not identify the admitted process bundle",
            ));
        }
        attempt.outcome.clone()?;
        match audit(&set.inner, set.inner.profile.eager_indices()) {
            Ok(observed) => Ok(observed),
            Err(error) => Err(attempt.retain_failure(error)),
        }
    }

    pub(super) fn verify_runtime(
        set: &ProcessLibrarySet,
        ort: &[OwnedLibrary],
    ) -> Result<(), LoadError> {
        verify_runtime_seen(set, ort).map(|_| ())
    }

    pub(super) fn observe_runtime(
        set: &ProcessLibrarySet,
        ort: &[OwnedLibrary],
    ) -> Result<NativeMappingObservation, LoadError> {
        verify_runtime_seen(set, ort).map(|seen| observation(set.inner.profile, &seen, true))
    }

    fn verify_runtime_seen(
        set: &ProcessLibrarySet,
        ort: &[OwnedLibrary],
    ) -> Result<BTreeSet<usize>, LoadError> {
        let mut latch = process().lock().map_err(|_| poisoned())?;
        let attempt = latch.attempt.as_mut().ok_or_else(poisoned)?;
        attempt.check_bundle(set.inner.fingerprint)?;
        if !Arc::ptr_eq(&attempt.pin, &set.inner) {
            return Err(LoadError::new(
                LoadErrorKind::DifferentBundle,
                "invalid native mapping token",
            ));
        }
        let checked =
            (|| {
                validate_names(ort, &ORT_LIBRARY_NAMES)?;
                let fingerprint = canonical_fingerprint(ort, &ORT_LIBRARY_NAMES)?;
                if let Some(previous) = &attempt.runtime {
                    if previous.fingerprint != fingerprint {
                        return Err(LoadError::new(
                            LoadErrorKind::DifferentBundle,
                            "a different ORT bundle was already pinned",
                        ));
                    }
                }
                let mut copies = Vec::with_capacity(ORT_LIBRARY_NAMES.len());
                for original in ort {
                    copies.push(FilePin {
                        path: original.path.clone(),
                        file: ManuallyDrop::new(original.file.try_clone().map_err(io_error)?),
                        digest: original.digest,
                        bytes: original.bytes,
                    });
                }
                copies.sort_by_key(|file| {
                    profile_index(&file.path, &ORT_LIBRARY_NAMES).unwrap_or(usize::MAX)
                });
                let expected = preflight_files(&copies, &ORT_LIBRARY_NAMES)?;
                let combined = attempt.pin.files.iter().chain(copies.iter()).try_fold(
                    0_u64,
                    |sum, file| {
                        sum.checked_add(file.bytes)
                            .filter(|total| *total <= MAX_TOTAL_BYTES)
                            .ok_or_else(|| {
                                LoadError::new(
                                    LoadErrorKind::ResourceLimit,
                                    "combined native and ORT bundle exceeds its byte bound",
                                )
                            })
                    },
                )?;
                let _ = combined;
                if attempt.runtime.is_none() {
                    attempt.runtime = Some(RuntimePin {
                        fingerprint,
                        files: copies,
                        expected,
                    });
                } else {
                    // 새 복제 FD는 native handle을 소유하지 않는다. immutable 동일 bundle 검사 후 닫는다.
                    close_temporary_files(copies);
                }
                audit(&set.inner, set.inner.profile.eager_indices())?;
                let runtime = attempt.runtime.as_ref().ok_or_else(poisoned)?;
                let current = preflight_files(&runtime.files, &ORT_LIBRARY_NAMES)?;
                if current.iter().zip(&runtime.expected).any(|(left, right)| {
                    left.path != right.path
                        || left.inode != right.inode
                        || left.dev_major != right.dev_major
                        || left.dev_minor != right.dev_minor
                }) {
                    return Err(LoadError::new(
                        LoadErrorKind::FileIdentity,
                        "pinned ORT file identity changed",
                    ));
                }
                let mut expected = set.inner.expected.clone();
                expected.extend(runtime.expected.iter().cloned());
                let mut required = set.inner.profile.eager_indices().to_vec();
                required.extend(NVIDIA_LOAD_ORDER.len()..expected.len());
                let seen = audit_entries_required(&expected, &required, &read_maps()?, true)?;
                Ok(seen)
            })();
        match checked {
            Ok(observed) => Ok(observed),
            Err(error) => Err(attempt.retain_failure(error)),
        }
    }

    fn sort_inputs(libraries: &mut [OwnedLibrary]) -> Result<(), LoadError> {
        validate_names(libraries, &NVIDIA_LOAD_ORDER)?;
        libraries.sort_by_key(|input| {
            profile_index(&input.path, &NVIDIA_LOAD_ORDER).unwrap_or(usize::MAX)
        });
        Ok(())
    }

    fn validate_names(libraries: &[OwnedLibrary], profile: &[&str]) -> Result<(), LoadError> {
        if libraries.len() > MAX_LIBRARY_COUNT || libraries.len() != profile.len() {
            return Err(LoadError::new(
                LoadErrorKind::InvalidBundle,
                "the complete supported NVIDIA profile is required",
            ));
        }
        let mut seen = BTreeSet::new();
        for input in libraries.iter() {
            let index = validate_declaration(&input.path, input.bytes, input.digest, profile)?;
            if !seen.insert(index) {
                return Err(LoadError::new(
                    LoadErrorKind::InvalidBundle,
                    "duplicate NVIDIA library name",
                ));
            }
        }
        Ok(())
    }

    pub(super) fn validate_runtime_profile(libraries: &[OwnedLibrary]) -> Result<(), LoadError> {
        validate_names(libraries, &ORT_LIBRARY_NAMES)
    }

    fn validate_declaration(
        path: &Path,
        bytes: u64,
        digest: [u8; 32],
        profile: &[&str],
    ) -> Result<usize, LoadError> {
        let index = profile_index(path, profile)?;
        let admitted = TRUSTED_NATIVE_PROFILE
            .iter()
            .find(|entry| entry.filename == profile[index])
            .ok_or_else(|| {
                LoadError::new(
                    LoadErrorKind::UntrustedProfile,
                    "library has no reviewed native admission",
                )
            })?;
        let matches = |entry: &TrustedNativeProfileEntry| {
            entry.filename == profile[index] && entry.bytes == bytes && entry.digest == digest
        };
        if !matches(admitted) && !matches(&TRUSTED_CUDART_12_8_90_WHEEL) {
            return Err(LoadError::new(
                LoadErrorKind::UntrustedProfile,
                "library declaration differs from the reviewed native profile",
            ));
        }
        Ok(index)
    }

    fn canonical_fingerprint(
        libraries: &[OwnedLibrary],
        profile: &[&str],
    ) -> Result<[u8; 32], LoadError> {
        validate_names(libraries, profile)?;
        let mut ordered: Vec<_> = libraries.iter().collect();
        ordered.sort_by_key(|input| profile_index(&input.path, profile).unwrap_or(usize::MAX));
        Ok(hash_inputs(ordered.len(), ordered.into_iter()))
    }

    fn input_fingerprint(libraries: &[OwnedLibrary]) -> [u8; 32] {
        hash_inputs(libraries.len(), libraries.iter().take(MAX_LIBRARY_COUNT))
    }

    fn hash_inputs<'a>(
        count: usize,
        libraries: impl Iterator<Item = &'a OwnedLibrary>,
    ) -> [u8; 32] {
        let mut hash = Sha256::new();
        hash.update(b"rz-native-loader-nvidia-v1\0");
        hash.update((count as u64).to_le_bytes());
        for input in libraries {
            let path = input.path.as_os_str().as_encoded_bytes();
            hash.update((path.len() as u64).to_le_bytes());
            hash.update(&path[..path.len().min(MAX_PATH_BYTES)]);
            hash.update(input.bytes.to_le_bytes());
            hash.update(input.digest);
        }
        hash.finalize().into()
    }

    fn close_temporary_files(files: Vec<FilePin>) {
        // native 초기화에 사용되지 않은 임시 복제만 닫는다. 원래 static pin/handle은 보존한다.
        for file in files {
            drop(ManuallyDrop::into_inner(file.file));
        }
    }

    #[cfg(test)]
    fn load_index(path: &Path) -> Result<usize, LoadError> {
        profile_index(path, &NVIDIA_LOAD_ORDER)
    }

    fn profile_index(path: &Path, profile: &[&str]) -> Result<usize, LoadError> {
        validate_path(path)?;
        let name = path.file_name().and_then(|name| name.to_str());
        profile
            .iter()
            .position(|candidate| Some(*candidate) == name)
            .ok_or_else(|| {
                LoadError::new(
                    LoadErrorKind::InvalidBundle,
                    "library name is outside the supported NVIDIA profile",
                )
            })
    }

    fn validate_path(path: &Path) -> Result<(), LoadError> {
        let text = path.to_str().ok_or_else(|| {
            LoadError::new(LoadErrorKind::InvalidBundle, "native path must be UTF-8")
        })?;
        if !path.is_absolute()
            || text.len() > MAX_PATH_BYTES
            || text
                .bytes()
                .any(|byte| matches!(byte, 0 | b'\n' | b'\r' | b'\\'))
            || text
                .split('/')
                .skip(1)
                .any(|part| part.is_empty() || part == "." || part == "..")
            || path.components().count() > MAX_PATH_DEPTH
        {
            return Err(LoadError::new(
                LoadErrorKind::InvalidBundle,
                "native path must be exact, absolute, bounded and unambiguous",
            ));
        }
        Ok(())
    }

    fn initialize(pin: &mut ProcessPin) -> Result<(), LoadError> {
        reject_ambient()?;
        reject_resident(&read_maps()?)?;
        preflight(pin)?;
        // Only a process that is actually about to execute a vendor native
        // constructor acquires the legacy regime. Invalid inputs do not prevent
        // the separate system-regime preflight from being used afterwards.
        // This atomic claim arbitrates concurrent legacy/system bootstrap.
        claim_native_mode(1)?;
        let mut required = Vec::with_capacity(pin.profile.eager_indices().len());
        for &index in pin.profile.eager_indices() {
            // SAFETY: 원래 bootstrap가 신뢰한 고정 NVIDIA profile만 admission한다.
            // 아래까지 모든 파일의 identity/hash/readonly capability를 검증했고,
            // dynamic lookup 대신 해당 absolute copied path를 직접 사용한다.
            // native constructor 자체의 soundness는 vendor trust 전제다. sandbox가 아니다.
            // 성공 handle은 즉시 ManuallyDrop으로 pin하여 dlclose/termination을 호출하지 않는다.
            let handle =
                unsafe { Library::open(Some(&pin.files[index].path), RTLD_NOW | RTLD_GLOBAL) }
                    .map_err(|original| {
                        LoadError::new(
                            LoadErrorKind::NativeLoad,
                            "native library initialization failed",
                        )
                        .with_diagnostic(original.to_string())
                    })?;
            pin.handles.push(ManuallyDrop::new(handle));
            required.push(index);
            audit(pin, &required)?;
        }
        audit(pin, pin.profile.eager_indices()).map(|_| ())
    }

    fn reject_ambient() -> Result<(), LoadError> {
        // 값은 출력·필드 보존·환경 수정 없이 즉시 버린다.
        if AMBIENT_VARIABLES
            .iter()
            .any(|name| std::env::var_os(name).is_some_and(|value| !value.is_empty()))
        {
            return Err(LoadError::new(
                LoadErrorKind::AmbientEnvironment,
                "ambient native injection or search settings are not permitted",
            ));
        }
        Ok(())
    }

    fn preflight(pin: &mut ProcessPin) -> Result<(), LoadError> {
        pin.expected = preflight_files(&pin.files, &NVIDIA_LOAD_ORDER)?;
        Ok(())
    }

    fn preflight_files(
        files: &[FilePin],
        profile: &[&str],
    ) -> Result<Vec<ExpectedMapping>, LoadError> {
        let mut total = 0_u64;
        for file in files {
            validate_declaration(&file.path, file.bytes, file.digest, profile)?;
            if file.bytes == 0 || file.bytes > MAX_LIBRARY_BYTES {
                return Err(LoadError::new(
                    LoadErrorKind::ResourceLimit,
                    "native library exceeds its finite byte limit",
                ));
            }
            total = total.checked_add(file.bytes).ok_or_else(|| {
                LoadError::new(LoadErrorKind::ResourceLimit, "native bundle size overflow")
            })?;
            if total > MAX_TOTAL_BYTES {
                return Err(LoadError::new(
                    LoadErrorKind::ResourceLimit,
                    "native bundle exceeds its total byte limit",
                ));
            }
        }
        let mut expected = Vec::with_capacity(files.len());
        for file in files {
            let metadata = validate_file(file)?;
            verify_digest(file)?;
            let after = validate_file(file)?;
            if !same_metadata(&metadata, &after) {
                return Err(LoadError::new(
                    LoadErrorKind::FileIdentity,
                    "native file changed during verification",
                ));
            }
            let (dev_major, dev_minor) = device_parts(metadata.dev());
            expected.push(ExpectedMapping {
                path: file.path.clone(),
                dev_major,
                dev_minor,
                inode: metadata.ino(),
            });
        }
        Ok(expected)
    }

    fn io_error(original: std::io::Error) -> LoadError {
        LoadError::new(
            LoadErrorKind::FileAccess,
            "native provenance input is unavailable",
        )
        .with_diagnostic(original.to_string())
    }

    fn validate_file(file: &FilePin) -> Result<Metadata, LoadError> {
        let mut prefix = PathBuf::new();
        for part in file.path.components() {
            prefix.push(part.as_os_str());
            let metadata = fs::symlink_metadata(&prefix).map_err(io_error)?;
            if metadata.file_type().is_symlink() {
                return Err(LoadError::new(
                    LoadErrorKind::FileIdentity,
                    "symlink components are not permitted for native copies",
                ));
            }
        }
        let descriptor = file.file.metadata().map_err(io_error)?;
        let path = fs::symlink_metadata(&file.path).map_err(io_error)?;
        if !descriptor.is_file()
            || !path.is_file()
            || !same_metadata(&descriptor, &path)
            || descriptor.len() != file.bytes
            || descriptor.mode() & 0o222 != 0
            || descriptor.nlink() != 1
        {
            return Err(LoadError::new(
                LoadErrorKind::FileIdentity,
                "native descriptor must identify the same readonly single-link copied file",
            ));
        }
        let fdinfo = read_bounded(
            &PathBuf::from(format!("/proc/self/fdinfo/{}", file.file.as_raw_fd())),
            MAX_FDINFO_BYTES,
        )?;
        if !readonly_descriptor(&fdinfo)? {
            return Err(LoadError::new(
                LoadErrorKind::FileIdentity,
                "native file capability must have been opened readonly",
            ));
        }
        Ok(descriptor)
    }

    fn same_metadata(left: &Metadata, right: &Metadata) -> bool {
        left.dev() == right.dev()
            && left.ino() == right.ino()
            && left.len() == right.len()
            && left.mode() == right.mode()
            && left.nlink() == right.nlink()
            && left.uid() == right.uid()
            && left.gid() == right.gid()
            && left.mtime() == right.mtime()
            && left.mtime_nsec() == right.mtime_nsec()
            && left.ctime() == right.ctime()
            && left.ctime_nsec() == right.ctime_nsec()
    }

    fn readonly_descriptor(bytes: &[u8]) -> Result<bool, LoadError> {
        let text = std::str::from_utf8(bytes).map_err(|_| {
            LoadError::new(LoadErrorKind::FileIdentity, "native fd flags are not UTF-8")
        })?;
        let mut flags = None;
        for line in text.lines() {
            if let Some(value) = line.strip_prefix("flags:") {
                if flags.is_some() {
                    return Err(LoadError::new(
                        LoadErrorKind::FileIdentity,
                        "duplicate native fd flags",
                    ));
                }
                flags = Some(u64::from_str_radix(value.trim(), 8).map_err(|_| {
                    LoadError::new(LoadErrorKind::FileIdentity, "invalid native fd flags")
                })?);
            }
        }
        flags.map(|value| value & 3 == 0).ok_or_else(|| {
            LoadError::new(LoadErrorKind::FileIdentity, "native fd flags are missing")
        })
    }

    fn verify_digest(file: &FilePin) -> Result<(), LoadError> {
        let mut digest = Sha256::new();
        let mut buffer = [0_u8; 64 * 1024];
        let mut count = 0_u64;
        let mut magic = [0_u8; 4];
        while count < file.bytes + 1 {
            let available = ((file.bytes + 1 - count).min(buffer.len() as u64)) as usize;
            // positional read는 caller/다른 clone의 공유 cursor를 변경하지 않는다.
            let read = file
                .file
                .read_at(&mut buffer[..available], count)
                .map_err(io_error)?;
            if read == 0 {
                break;
            }
            if count < magic.len() as u64 {
                let start = count as usize;
                let prefix = read.min(magic.len() - start);
                magic[start..start + prefix].copy_from_slice(&buffer[..prefix]);
            }
            count += read as u64;
            digest.update(&buffer[..read]);
        }
        let actual: [u8; 32] = digest.finalize().into();
        if count != file.bytes || actual != file.digest || magic != *b"\x7fELF" {
            return Err(LoadError::new(
                LoadErrorKind::FileIdentity,
                "native bytes do not match the admitted ELF manifest",
            ));
        }
        Ok(())
    }

    #[derive(Debug)]
    pub(super) struct MapEntry {
        pub(super) dev_major: u64,
        pub(super) dev_minor: u64,
        pub(super) inode: u64,
        pub(super) path: Option<PathBuf>,
        pub(super) deleted: bool,
    }

    fn invalid_maps() -> LoadError {
        LoadError::new(
            LoadErrorKind::InvalidMaps,
            "process mapping evidence is malformed or ambiguous",
        )
    }

    fn read_bounded(path: &Path, limit: usize) -> Result<Vec<u8>, LoadError> {
        let mut result = Vec::new();
        File::open(path)
            .map_err(io_error)?
            .take(limit as u64 + 1)
            .read_to_end(&mut result)
            .map_err(io_error)?;
        if result.len() > limit {
            return Err(LoadError::new(
                LoadErrorKind::ResourceLimit,
                "process provenance evidence exceeds its bound",
            ));
        }
        Ok(result)
    }

    pub(super) fn read_maps() -> Result<Vec<MapEntry>, LoadError> {
        parse_maps(&read_bounded(Path::new("/proc/self/maps"), MAX_MAPS_BYTES)?)
    }

    fn parse_maps(bytes: &[u8]) -> Result<Vec<MapEntry>, LoadError> {
        if bytes.len() > MAX_MAPS_BYTES {
            return Err(LoadError::new(
                LoadErrorKind::ResourceLimit,
                "process maps exceeds its byte bound",
            ));
        }
        let text = std::str::from_utf8(bytes).map_err(|_| invalid_maps())?;
        let mut entries = Vec::new();
        for line in text.lines() {
            if line.is_empty() {
                continue;
            }
            // 5개 고정 필드 뒤의 pathname은 공백을 포함할 수 있으므로 split_whitespace로 분해하지 않는다.
            let mut remainder = line;
            let mut fields = [""; 5];
            for field in &mut fields {
                remainder = remainder.trim_start_matches([' ', '\t']);
                let end = remainder.find([' ', '\t']).unwrap_or(remainder.len());
                if end == 0 {
                    return Err(invalid_maps());
                }
                *field = &remainder[..end];
                remainder = &remainder[end..];
            }
            let (start, end) = fields[0].split_once('-').ok_or_else(invalid_maps)?;
            let start = u64::from_str_radix(start, 16).map_err(|_| invalid_maps())?;
            let end = u64::from_str_radix(end, 16).map_err(|_| invalid_maps())?;
            let permissions = fields[1].as_bytes();
            if start >= end
                || permissions.len() != 4
                || !matches!(permissions[0], b'r' | b'-')
                || !matches!(permissions[1], b'w' | b'-')
                || !matches!(permissions[2], b'x' | b'-')
                || !matches!(permissions[3], b'p' | b's')
                || u64::from_str_radix(fields[2], 16).is_err()
            {
                return Err(invalid_maps());
            }
            let (major, minor) = fields[3].split_once(':').ok_or_else(invalid_maps)?;
            let dev_major = u64::from_str_radix(major, 16).map_err(|_| invalid_maps())?;
            let dev_minor = u64::from_str_radix(minor, 16).map_err(|_| invalid_maps())?;
            let inode = fields[4].parse().map_err(|_| invalid_maps())?;
            let pathname = remainder.trim_start_matches([' ', '\t']);
            let deleted = pathname.ends_with(" (deleted)");
            let pathname = pathname.strip_suffix(" (deleted)").unwrap_or(pathname);
            let path = if pathname.is_empty() || pathname.starts_with('[') {
                None
            } else {
                let path = PathBuf::from(pathname);
                validate_path(&path).map_err(|_| invalid_maps())?;
                Some(path)
            };
            entries.push(MapEntry {
                dev_major,
                dev_minor,
                inode,
                path,
                deleted,
            });
        }
        Ok(entries)
    }

    pub(super) fn is_nvidia_runtime(name: &str) -> bool {
        [
            "libcudnn",
            "libcublas",
            "libcurand",
            "libcufft",
            "libcudart",
            "libnvrtc",
            "libnvJitLink",
            "libcusolver",
            "libcusparse",
            "libcupti",
            "libnpp",
            "libnvblas",
            "libnccl",
        ]
        .iter()
        .any(|prefix| name.starts_with(prefix))
    }

    fn reject_resident(entries: &[MapEntry]) -> Result<(), LoadError> {
        for entry in entries {
            let name = entry
                .path
                .as_ref()
                .and_then(|path| path.file_name())
                .and_then(|name| name.to_str());
            if name
                .is_some_and(|name| is_nvidia_runtime(name) || name.starts_with("libonnxruntime"))
            {
                return Err(LoadError::new(
                    LoadErrorKind::ForeignMapping,
                    "NVIDIA runtime or ORT was mapped before native admission",
                ));
            }
        }
        Ok(())
    }

    pub(super) fn device_parts(device: u64) -> (u64, u64) {
        (
            ((device >> 8) & 0xfff) | ((device >> 32) & 0xffff_f000),
            (device & 0xff) | ((device >> 12) & 0xffff_ff00),
        )
    }

    fn audit(pin: &ProcessPin, required: &[usize]) -> Result<BTreeSet<usize>, LoadError> {
        reject_ambient()?;
        for (file, expected) in pin.files.iter().zip(&pin.expected) {
            let metadata = validate_file(file)?;
            let (major, minor) = device_parts(metadata.dev());
            if expected.path != file.path
                || expected.inode != metadata.ino()
                || expected.dev_major != major
                || expected.dev_minor != minor
            {
                return Err(LoadError::new(
                    LoadErrorKind::FileIdentity,
                    "admitted native file identity changed",
                ));
            }
        }
        audit_entries_required(&pin.expected, required, &read_maps()?, false)
    }

    fn observation(
        profile: NativeLoadingProfile,
        seen: &BTreeSet<usize>,
        include_ort: bool,
    ) -> NativeMappingObservation {
        NativeMappingObservation {
            profile,
            declared_nvidia_files: NVIDIA_LOAD_ORDER.len(),
            required_nvidia_files: profile
                .eager_indices()
                .iter()
                .map(|&index| NVIDIA_LOAD_ORDER[index])
                .collect(),
            mapped_nvidia_files: NVIDIA_LOAD_ORDER
                .iter()
                .enumerate()
                .filter(|(index, _)| seen.contains(index))
                .map(|(_, &name)| name)
                .collect(),
            deferred_nvidia_not_mapped: NVIDIA_LOAD_ORDER
                .iter()
                .enumerate()
                .filter(|(index, _)| {
                    !profile.eager_indices().contains(index) && !seen.contains(index)
                })
                .map(|(_, &name)| name)
                .collect(),
            mapped_ort_files: include_ort.then(|| {
                ORT_LIBRARY_NAMES
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| seen.contains(&(NVIDIA_LOAD_ORDER.len() + index)))
                    .map(|(_, &name)| name)
                    .collect()
            }),
        }
    }

    #[cfg(test)]
    fn audit_entries(
        expected: &[ExpectedMapping],
        required: usize,
        entries: &[MapEntry],
        include_ort: bool,
    ) -> Result<(), LoadError> {
        let required: Vec<_> = (0..required).collect();
        audit_entries_required(expected, &required, entries, include_ort).map(|_| ())
    }

    fn audit_entries_required(
        expected: &[ExpectedMapping],
        required: &[usize],
        entries: &[MapEntry],
        include_ort: bool,
    ) -> Result<BTreeSet<usize>, LoadError> {
        if required.len() > expected.len() || required.iter().any(|&index| index >= expected.len())
        {
            return Err(LoadError::new(
                LoadErrorKind::InvalidBundle,
                "mapping requirement differs from the complete pinned profile",
            ));
        }
        let mut seen = BTreeSet::new();
        for entry in entries {
            let Some(path) = &entry.path else {
                continue;
            };
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                return Err(invalid_maps());
            };
            if !(is_nvidia_runtime(name) || include_ort && name.starts_with("libonnxruntime")) {
                // libcuda.so.1과 libnvidia-*는 별도의 host driver provenance 축이다.
                continue;
            }
            let Some(index) = expected.iter().position(|mapping| mapping.path == *path) else {
                return Err(LoadError::new(
                    LoadErrorKind::ForeignMapping,
                    "foreign NVIDIA runtime mapping detected",
                ));
            };
            let mapping = &expected[index];
            if entry.deleted
                || entry.inode == 0
                || entry.inode != mapping.inode
                || entry.dev_major != mapping.dev_major
                || entry.dev_minor != mapping.dev_minor
            {
                return Err(LoadError::new(
                    LoadErrorKind::MappingIdentity,
                    "NVIDIA mapped file identity differs from its pin",
                ));
            }
            seen.insert(index);
        }
        if required.iter().any(|index| !seen.contains(index)) {
            return Err(LoadError::new(
                LoadErrorKind::MissingMapping,
                "an expected NVIDIA library is not mapped",
            ));
        }
        Ok(seen)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn expected() -> Vec<ExpectedMapping> {
            vec![ExpectedMapping {
                path: PathBuf::from("/private native/libcudart.so.12"),
                dev_major: 8,
                dev_minor: 2,
                inode: 19,
            }]
        }

        #[test]
        fn maps_preserve_path_spaces_and_device_inode_identity() {
            let entries = parse_maps(b"1000-2000 r-xp 0000 08:02 19 /private native/libcudart.so.12\n2000-3000 rw-p 1000 00:00 0\n").unwrap();
            assert_eq!(entries[0].path.as_ref().unwrap(), &expected()[0].path);
            assert_eq!(entries[0].inode, 19);
            assert!(entries[1].path.is_none());
            assert!(audit_entries(&expected(), 1, &entries, false).is_ok());
        }

        fn whole_expected() -> Vec<ExpectedMapping> {
            NVIDIA_LOAD_ORDER
                .iter()
                .chain(&ORT_LIBRARY_NAMES)
                .enumerate()
                .map(|(index, name)| ExpectedMapping {
                    path: PathBuf::from(format!("/private native/{name}")),
                    dev_major: 8,
                    dev_minor: 2,
                    inode: 19 + index as u64,
                })
                .collect()
        }

        fn mapped(expected: &[ExpectedMapping], indices: &[usize]) -> Vec<MapEntry> {
            indices
                .iter()
                .map(|&index| {
                    let pin = &expected[index];
                    MapEntry {
                        dev_major: pin.dev_major,
                        dev_minor: pin.dev_minor,
                        inode: pin.inode,
                        path: Some(pin.path.clone()),
                        deleted: false,
                    }
                })
                .collect()
        }

        #[test]
        fn shim_requires_nonprefix_roots_but_exact_deferred_subset_is_optional() {
            let expected = whole_expected();
            let profile = NativeLoadingProfile::CuDnnShimLazyV1;
            let roots = profile.eager_indices();
            let mut entries = mapped(&expected, roots);
            let seen = audit_entries_required(&expected[..16], roots, &entries, false).unwrap();
            let observed = observation(profile, &seen, false);
            assert_eq!(observed.required_nvidia_files.len(), 9);
            assert_eq!(observed.mapped_nvidia_files.len(), 9);
            assert_eq!(observed.deferred_nvidia_not_mapped.len(), 7);
            assert_eq!(observed.mapped_ort_files, None);
            assert!(observed.required_nvidia_files.contains(&"libcudnn.so.9"));
            entries.extend(mapped(&expected, &[8, 12]));
            let seen = audit_entries_required(&expected[..16], roots, &entries, false).unwrap();
            let observed = observation(profile, &seen, false);
            assert_eq!(observed.mapped_nvidia_files.len(), 11);
            assert_eq!(observed.deferred_nvidia_not_mapped.len(), 5);
            assert_eq!(
                audit_entries_required(
                    &expected[..16],
                    NativeLoadingProfile::EagerCuda12Cudnn9V1.eager_indices(),
                    &entries,
                    false,
                )
                .unwrap_err()
                .kind,
                LoadErrorKind::MissingMapping
            );
            entries.remove(8); // facade, not the eighth prefix image, is mandatory
            assert_eq!(
                audit_entries_required(&expected[..16], roots, &entries, false)
                    .unwrap_err()
                    .kind,
                LoadErrorKind::MissingMapping
            );
        }

        #[test]
        fn shim_deferred_mapping_still_rejects_foreign_deleted_and_wrong_identity() {
            let expected = whole_expected();
            let roots = NativeLoadingProfile::CuDnnShimLazyV1.eager_indices();
            for bad in 0..4 {
                let mut entries = mapped(&expected, roots);
                let mut deferred = mapped(&expected, &[12]).pop().unwrap();
                match bad {
                    0 => {
                        deferred.path =
                            Some(PathBuf::from("/ambient/libcudnn_engines_precompiled.so.9"))
                    }
                    1 => deferred.deleted = true,
                    2 => deferred.inode += 1,
                    _ => deferred.dev_minor += 1,
                }
                entries.push(deferred);
                assert!(audit_entries_required(&expected[..16], roots, &entries, false).is_err());
            }
        }

        #[test]
        fn shim_full_audit_adds_three_mandatory_ort_images() {
            let expected = whole_expected();
            let profile = NativeLoadingProfile::CuDnnShimLazyV1;
            let mut required = profile.eager_indices().to_vec();
            required.extend(16..19);
            let entries = mapped(&expected, &required);
            let seen = audit_entries_required(&expected, &required, &entries, true).unwrap();
            let observed = observation(profile, &seen, true);
            assert_eq!(observed.mapped_ort_files.as_ref().unwrap().len(), 3);
            assert_eq!(observed.deferred_nvidia_not_mapped.len(), 7);
            for omitted in 9..12 {
                let mut incomplete = mapped(&expected, &required);
                incomplete.remove(omitted);
                assert_eq!(
                    audit_entries_required(&expected, &required, &incomplete, true)
                        .unwrap_err()
                        .kind,
                    LoadErrorKind::MissingMapping
                );
            }
        }

        #[test]
        fn mappings_reject_foreign_deleted_wrong_inode_and_missing_pins() {
            for line in [
                "1000-2000 r-xp 0 08:02 19 /ambient/libcudart.so.12\n",
                "1000-2000 r-xp 0 08:02 19 /private native/libcudart.so.12 (deleted)\n",
                "1000-2000 r-xp 0 08:02 20 /private native/libcudart.so.12\n",
                "1000-2000 r-xp 0 08:03 19 /private native/libcudart.so.12\n",
                "1000-2000 r-xp 0 08:02 19 /ambient/libcudnn.so.8\n",
            ] {
                assert!(
                    audit_entries(&expected(), 1, &parse_maps(line.as_bytes()).unwrap(), false)
                        .is_err()
                );
            }
            assert_eq!(
                audit_entries(&expected(), 1, &[], false).unwrap_err().kind,
                LoadErrorKind::MissingMapping
            );
            let driver = parse_maps(b"1000-2000 r-xp 0 08:02 90 /usr/lib/libcuda.so.1\n").unwrap();
            assert!(reject_resident(&driver).is_ok());
        }

        #[test]
        fn whole_runtime_audit_requires_exact_ort_mapping_and_rejects_extras() {
            let mut both = expected();
            both.push(ExpectedMapping {
                path: PathBuf::from("/private native/libonnxruntime.so.1.22.0"),
                dev_major: 8,
                dev_minor: 2,
                inode: 20,
            });
            let admitted = b"1000-2000 r-xp 0 08:02 19 /private native/libcudart.so.12\n2000-3000 r-xp 0 08:02 20 /private native/libonnxruntime.so.1.22.0\n";
            let entries = parse_maps(admitted).unwrap();
            assert!(audit_entries(&both, 2, &entries, true).is_ok());
            assert!(reject_resident(&entries).is_err());
            assert_eq!(
                audit_entries(&both, 2, &entries[..1], true)
                    .unwrap_err()
                    .kind,
                LoadErrorKind::MissingMapping
            );
            let mut extra = entries;
            extra.extend(
                parse_maps(b"3000-4000 r-xp 0 08:02 21 /ambient/libonnxruntime.so.1.21.0\n")
                    .unwrap(),
            );
            assert_eq!(
                audit_entries(&both, 2, &extra, true).unwrap_err().kind,
                LoadErrorKind::ForeignMapping
            );
        }

        #[test]
        fn maps_reject_malformed_ambiguous_and_non_utf8_evidence() {
            for line in [
                b"2000-1000 r-xp 0 08:02 19 /a/libcudart.so.12\n".as_slice(),
                b"1000-2000 garbage 0 08:02 19 /a/libcudart.so.12\n",
                b"1000-2000 r-xp 0 08:02 19 /a\\012b/libcudart.so.12\n",
                b"1000-2000 r-xp 0 08:02 19 /a/../libcudart.so.12\n",
                b"1000-2000 r-xp 0 08:02 19 /a/\xfflibcudart.so.12\n",
            ] {
                assert!(parse_maps(line).is_err());
            }
            assert_eq!(
                parse_maps(&vec![b'x'; MAX_MAPS_BYTES + 1])
                    .unwrap_err()
                    .kind,
                LoadErrorKind::ResourceLimit
            );
        }

        #[test]
        fn admission_paths_are_exact_absolute_and_closed_profile() {
            assert!(load_index(Path::new("/private native/libcudart.so.12")).is_ok());
            for path in [
                "libcudart.so.12",
                "/a/./libcudart.so.12",
                "/a//libcudart.so.12",
                "/a/libcudart.so.11",
                "/a/libcuda.so.1",
            ] {
                assert!(load_index(Path::new(path)).is_err());
            }
            assert!(validate_path(Path::new(&format!("/{}", "a".repeat(MAX_PATH_BYTES)))).is_err());
        }

        #[test]
        fn descriptor_flags_require_readonly_and_unambiguous_evidence() {
            assert!(readonly_descriptor(b"pos:\t0\nflags:\t02100000\n").unwrap());
            assert!(!readonly_descriptor(b"flags:\t02100002\n").unwrap());
            assert!(readonly_descriptor(b"flags:\t0\nflags:\t0\n").is_err());
            assert!(readonly_descriptor(b"pos:\t0\n").is_err());
        }

        #[test]
        fn exact_name_does_not_admit_invented_hash_or_size() {
            let accepted = TRUSTED_NATIVE_PROFILE[0];
            let path = Path::new("/private native/libcudart.so.12");
            assert!(
                validate_declaration(path, accepted.bytes, accepted.digest, &NVIDIA_LOAD_ORDER)
                    .is_ok()
            );
            for (bytes, digest) in [
                (accepted.bytes, [0; 32]),
                (accepted.bytes + 1, accepted.digest),
                (4, accepted.digest),
            ] {
                assert_eq!(
                    validate_declaration(path, bytes, digest, &NVIDIA_LOAD_ORDER)
                        .unwrap_err()
                        .kind,
                    LoadErrorKind::UntrustedProfile
                );
            }
            let ort = TRUSTED_NATIVE_PROFILE[16];
            assert_eq!(
                validate_declaration(path, ort.bytes, ort.digest, &NVIDIA_LOAD_ORDER)
                    .unwrap_err()
                    .kind,
                LoadErrorKind::UntrustedProfile
            );
        }

        #[test]
        fn reviewed_cudart_wheel_preserves_legacy_and_rejects_other_declarations() {
            let wheel = TRUSTED_CUDART_12_8_90_WHEEL;
            let path = Path::new("/private native/libcudart.so.12");
            for entry in [TRUSTED_NATIVE_PROFILE[0], wheel] {
                assert!(
                    validate_declaration(path, entry.bytes, entry.digest, &NVIDIA_LOAD_ORDER)
                        .is_ok()
                );
            }
            let mut changed = wheel.digest;
            changed[0] ^= 1;
            for (name, bytes, digest) in [
                (path, wheel.bytes + 1, wheel.digest),
                (path, wheel.bytes, changed),
                (
                    Path::new("/private native/libnvJitLink.so.12"),
                    wheel.bytes,
                    wheel.digest,
                ),
            ] {
                assert_eq!(
                    validate_declaration(name, bytes, digest, &NVIDIA_LOAD_ORDER)
                        .unwrap_err()
                        .kind,
                    LoadErrorKind::UntrustedProfile
                );
            }
            let mut declared: Vec<_> = TRUSTED_NATIVE_PROFILE[..16]
                .iter()
                .map(|entry| OwnedLibrary {
                    path: PathBuf::from(format!("/private native/{}", entry.filename)),
                    file: File::open("/dev/null").unwrap(),
                    digest: entry.digest,
                    bytes: entry.bytes,
                })
                .collect();
            declared[0].digest = wheel.digest;
            assert!(sort_inputs(&mut declared).is_ok());
            declared.pop();
            assert_eq!(
                sort_inputs(&mut declared).unwrap_err().kind,
                LoadErrorKind::InvalidBundle
            );
        }

        #[test]
        fn ort_declaration_preflight_and_fingerprint_are_order_independent() {
            // File은 선언 preflight의 내용 검증 증거가 아니다. native 코드는 호출하지 않는다.
            let mut declared: Vec<_> = TRUSTED_NATIVE_PROFILE[16..]
                .iter()
                .map(|entry| OwnedLibrary {
                    path: PathBuf::from(format!("/private native/{}", entry.filename)),
                    file: File::open("/dev/null").unwrap(),
                    digest: entry.digest,
                    bytes: entry.bytes,
                })
                .collect();
            validate_runtime_profile(&declared).unwrap();
            let first = canonical_fingerprint(&declared, &ORT_LIBRARY_NAMES).unwrap();
            declared.rotate_left(1);
            assert_eq!(
                canonical_fingerprint(&declared, &ORT_LIBRARY_NAMES).unwrap(),
                first
            );
            declared.reverse();
            assert_eq!(
                canonical_fingerprint(&declared, &ORT_LIBRARY_NAMES).unwrap(),
                first
            );
            declared[0].digest[0] ^= 1;
            assert_eq!(
                validate_runtime_profile(&declared).unwrap_err().kind,
                LoadErrorKind::UntrustedProfile
            );
        }

        #[test]
        fn shim_keeps_complete_deferred_declarations_before_any_native_constructor() {
            let mut declared: Vec<_> = TRUSTED_NATIVE_PROFILE[..16]
                .iter()
                .map(|entry| OwnedLibrary {
                    path: PathBuf::from(format!("/private native/{}", entry.filename)),
                    file: File::open("/dev/null").unwrap(),
                    digest: entry.digest,
                    bytes: entry.bytes,
                })
                .collect();
            sort_inputs(&mut declared).unwrap();
            // Only declaration seams are exercised: /dev/null is never passed
            // to native loading or treated as content-hash verification.
            declared[12].digest[0] ^= 1;
            assert_eq!(
                sort_inputs(&mut declared).unwrap_err().kind,
                LoadErrorKind::UntrustedProfile
            );
            declared.remove(12);
            assert_eq!(
                sort_inputs(&mut declared).unwrap_err().kind,
                LoadErrorKind::InvalidBundle
            );
        }

        #[test]
        fn identical_bundle_cannot_switch_loading_policy_or_clear_first_failure() {
            let fingerprint = [7; 32];
            let mut attempt = Attempt {
                fingerprint,
                pin: Arc::new(ProcessPin::new(
                    fingerprint,
                    NativeLoadingProfile::CuDnnShimLazyV1,
                    Vec::new(),
                )),
                outcome: Ok(()),
                runtime: None,
            };
            attempt.check_bundle(fingerprint).unwrap();
            attempt
                .check_profile(NativeLoadingProfile::CuDnnShimLazyV1)
                .unwrap();
            let first = attempt
                .check_profile(NativeLoadingProfile::EagerCuda12Cudnn9V1)
                .unwrap_err();
            assert_eq!(first.kind, LoadErrorKind::DifferentBundle);
            assert_eq!(
                attempt
                    .check_profile(NativeLoadingProfile::CuDnnShimLazyV1)
                    .unwrap_err()
                    .detail,
                first.detail
            );
            assert_eq!(
                attempt.check_bundle(fingerprint).unwrap_err().detail,
                first.detail
            );
        }

        #[test]
        fn failure_latch_preserves_first_error_and_rejects_another_bundle() {
            let fingerprint = [7; 32];
            let first = LoadError::new(LoadErrorKind::NativeLoad, "first failure")
                .with_diagnostic("bounded original cause".to_owned());
            let pin = Arc::new(ProcessPin::new(
                fingerprint,
                NativeLoadingProfile::EagerCuda12Cudnn9V1,
                Vec::new(),
            ));
            let mut attempt = Attempt {
                fingerprint,
                pin,
                outcome: Ok(()),
                runtime: None,
            };
            assert_eq!(
                attempt.retain_failure(first.clone()).diagnostic(),
                first.diagnostic()
            );
            let later = LoadError::new(LoadErrorKind::MappingIdentity, "later failure");
            assert_eq!(
                attempt.retain_failure(later).kind,
                LoadErrorKind::NativeLoad
            );
            assert_eq!(
                attempt.check_bundle([8; 32]).unwrap_err().kind,
                LoadErrorKind::DifferentBundle
            );
            assert_eq!(
                attempt.check_bundle(fingerprint).unwrap_err().diagnostic(),
                first.diagnostic()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "linux")]
    #[test]
    fn rejected_legacy_preflight_does_not_claim_system_regime() {
        const CHILD: &str = "RZ_SYSTEM_MODE_PREFLIGHT_TEST_CHILD";
        if std::env::var_os(CHILD).is_some() {
            // An empty bundle fails before any native constructor can run.
            assert!(LibrarySet::load(Vec::new()).is_err());
            assert_eq!(NATIVE_OWNERSHIP_MODE.load(Ordering::Acquire), 0);
            assert!(claim_native_mode(2).is_ok());
            return;
        }
        // Isolate the process-global native latch from unrelated test cases.
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "tests::rejected_legacy_preflight_does_not_claim_system_regime",
            ])
            .env(CHILD, "1")
            .status()
            .unwrap();
        assert!(status.success());
    }

    #[test]
    fn native_diagnostic_is_bounded_and_not_in_default_formatting() {
        let original = format!("private-path/{}", "한".repeat(1000));
        let error = LoadError::new(LoadErrorKind::NativeLoad, "native load failed")
            .with_diagnostic(original.clone());
        let saved = error.diagnostic().unwrap();
        assert!(saved.len() <= MAX_DIAGNOSTIC_BYTES);
        assert!(error.diagnostic_truncated());
        assert!(original.starts_with(saved));
        assert!(!format!("{error}").contains("private-path"));
        assert!(!format!("{error:?}").contains("private-path"));
        let complete = LoadError::new(LoadErrorKind::NativeLoad, "native load failed")
            .with_diagnostic("short cause".to_owned());
        assert!(!complete.diagnostic_truncated());
    }

    #[test]
    fn trusted_profile_is_exactly_the_closed_16_plus_3() {
        assert_eq!(
            TRUSTED_NATIVE_PROFILE.len(),
            NVIDIA_LOAD_ORDER.len() + ORT_LIBRARY_NAMES.len()
        );
        for (entry, name) in TRUSTED_NATIVE_PROFILE
            .iter()
            .zip(NVIDIA_LOAD_ORDER.iter().chain(&ORT_LIBRARY_NAMES))
        {
            assert_eq!(entry.filename, *name);
            assert!(entry.bytes > 0 && entry.bytes <= MAX_LIBRARY_BYTES);
            assert_ne!(entry.digest, [0; 32]);
        }
        assert!(
            TRUSTED_NATIVE_PROFILE
                .iter()
                .map(|entry| entry.bytes)
                .sum::<u64>()
                <= MAX_TOTAL_BYTES
        );
    }
}
