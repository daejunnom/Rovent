//! System-provided native library identity without per-run library copies.
//!
//! This module does not load native code or install packages. The caller must
//! select its system library paths before ORT is initialised, then own ORT's
//! session and native handles until physical GPU completion. An ELF header and
//! a mapping observation do not prove ORT/provider ABI compatibility, GPU
//! execution, numerical correctness, or safe completion. Those are separate
//! checks by the backend owner.

use super::{LoadError, LoadErrorKind, claim_native_mode, linux};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, Metadata, OpenOptions};
use std::io::Read;
use std::os::fd::AsRawFd;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// A runtime selected from the host installation, not a copied artifact.
/// Paths may contain ordinary system symlinks; the resolved regular file and
/// its open readonly descriptor are pinned for the lifetime of the process.
#[derive(Clone, Debug)]
pub struct SystemLibraryRequest {
    pub role: String,
    pub path: PathBuf,
    /// False is appropriate for an optional dependency loaded lazily.
    pub required_mapping: bool,
}

/// Caller-registered finite admission limits. Elapsed time is checked at read
/// boundaries; a single blocking OS read is not forcibly interrupted.
#[derive(Clone, Copy, Debug)]
pub struct SystemCaptureLimits {
    pub max_libraries: usize,
    pub max_file_bytes: u64,
    pub max_total_bytes: u64,
    pub max_elapsed: Duration,
}

/// What was actually read from one system library, not a version assertion.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SystemLibraryIdentity {
    pub role: String,
    pub canonical_path: PathBuf,
    pub sha256: [u8; 32],
    pub bytes: u64,
    pub device: u64,
    pub inode: u64,
    pub elf_class: u8,
    pub elf_machine: u16,
    pub required_mapping: bool,
    /// Wall time to inspect and hash this file during the first capture.
    /// A reused token retains the original observation; it is not a new timing.
    pub first_capture_elapsed_ns: u64,
}

/// An observation of this process only. A missing optional mapping is not
/// represented as evidence that the library cannot be needed later.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SystemMappingAvailability {
    Observed,
    Unavailable,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SystemMappingObservation {
    pub identity_sha256: [u8; 32],
    /// An unavailable process-maps observation is neither an empty map nor
    /// evidence that the declared libraries are absent.
    pub availability: SystemMappingAvailability,
    /// Whether the pre-capture mapping state was actually observed. A later
    /// successful read cannot retroactively establish that earlier observation.
    pub preexisting_mapping_observed: bool,
    /// Native images seen before capture. Even an exact file match does not
    /// prove that this owner controlled the earlier load or its lifetime.
    pub preexisting_native_mappings: Vec<String>,
    pub mapped_roles: Vec<String>,
    pub missing_required_roles: Vec<String>,
    pub missing_optional_roles: Vec<String>,
    /// Basenames only; these mappings were not part of the captured inputs.
    pub unregistered_native_mappings: Vec<String>,
    /// Mapping alone never proves provider execution or device completion.
    pub gpu_provider_verified: bool,
    pub gpu_completion_verified: bool,
}

struct LibraryPin {
    file: File,
    metadata: Metadata,
    identity: SystemLibraryIdentity,
}

struct Inner {
    identity_sha256: [u8; 32],
    libraries: Vec<LibraryPin>,
    preexisting_mapping_observed: bool,
    preexisting_native_mappings: Vec<String>,
}

#[derive(Default)]
struct ProcessState {
    owner: Option<Arc<Inner>>,
    failure: Option<LoadError>,
}

fn state() -> &'static Mutex<ProcessState> {
    static STATE: OnceLock<Mutex<ProcessState>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(ProcessState::default()))
}

fn error(kind: LoadErrorKind, detail: &'static str) -> LoadError {
    LoadError::new(kind, detail)
}

fn poison() -> LoadError {
    error(
        LoadErrorKind::ProcessPoisoned,
        "system runtime owner is poisoned",
    )
}

/// A token for the process-global readonly descriptors. Clones do not create
/// copies, and dropping every token does not release the process-global pins.
#[derive(Clone)]
pub struct SystemRuntime {
    inner: Arc<Inner>,
}

impl SystemRuntime {
    /// Capture before loading CUDA/ORT. This validates actual bytes and ELF
    /// architecture without requiring fixed build hashes or copying any file.
    /// Successful capture is not permission to run an untrusted library.
    pub fn capture(
        mut requests: Vec<SystemLibraryRequest>,
        limits: SystemCaptureLimits,
    ) -> Result<Self, LoadError> {
        if limits.max_libraries == 0
            || limits.max_file_bytes == 0
            || limits.max_total_bytes == 0
            || limits.max_elapsed.is_zero()
            || requests.is_empty()
            || requests.len() > limits.max_libraries
        {
            return Err(error(
                LoadErrorKind::ResourceLimit,
                "invalid or exceeded system library limits",
            ));
        }
        let deadline = Instant::now()
            .checked_add(limits.max_elapsed)
            .ok_or_else(|| {
                error(
                    LoadErrorKind::ResourceLimit,
                    "system capture deadline overflow",
                )
            })?;
        if ["LD_PRELOAD", "LD_AUDIT", "CUDA_INJECTION64_PATH"]
            .iter()
            .any(|key| std::env::var_os(key).is_some_and(|value| !value.is_empty()))
        {
            return Err(error(
                LoadErrorKind::AmbientEnvironment,
                "injected native environment is present",
            ));
        }
        requests.sort_by(|left, right| left.role.cmp(&right.role));
        {
            let mut guard = state().lock().map_err(|_| poison())?;
            if let Some(failure) = &guard.failure {
                return Err(failure.clone());
            }
            if let Some(existing) = guard.owner.as_ref().map(Arc::clone) {
                if existing.libraries.len() != requests.len() {
                    return Err(error(
                        LoadErrorKind::DifferentBundle,
                        "system runtime changed within the process",
                    ));
                }
                let mut remaining = limits.max_total_bytes;
                for (request, pin) in requests.iter().zip(&existing.libraries) {
                    check_deadline(deadline)?;
                    if pin.identity.bytes > limits.max_file_bytes || pin.identity.bytes > remaining
                    {
                        return Err(error(
                            LoadErrorKind::ResourceLimit,
                            "system library byte limit exceeded",
                        ));
                    }
                    remaining -= pin.identity.bytes;
                    if !request.path.is_absolute() {
                        return Err(error(
                            LoadErrorKind::FileIdentity,
                            "system library path must be absolute",
                        ));
                    }
                    let path = fs::canonicalize(&request.path).map_err(|_| {
                        error(
                            LoadErrorKind::FileAccess,
                            "system library cannot be resolved",
                        )
                    })?;
                    if request.role != pin.identity.role
                        || request.required_mapping != pin.identity.required_mapping
                        || path != pin.identity.canonical_path
                    {
                        return Err(error(
                            LoadErrorKind::DifferentBundle,
                            "system runtime changed within the process",
                        ));
                    }
                    if let Err(failure) = check_still_named(pin) {
                        guard.failure = Some(failure.clone());
                        return Err(failure);
                    }
                }
                // The original open file descriptors and actual hash remain
                // pinned. No new read/hash/copy occurs for same-owner reuse.
                return Ok(Self { inner: existing });
            }
        }
        let mut roles = BTreeSet::new();
        let mut files = BTreeSet::new();
        let mut libraries = Vec::with_capacity(requests.len());
        let mut remaining = limits.max_total_bytes;
        for request in requests {
            check_deadline(deadline)?;
            if request.role.is_empty()
                || request.role.len() > 128
                || !request
                    .role
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"-_.".contains(&c))
                || !roles.insert(request.role.clone())
            {
                return Err(error(
                    LoadErrorKind::InvalidBundle,
                    "invalid or duplicate system library role",
                ));
            }
            let pin = capture_one(request, limits.max_file_bytes, &mut remaining, deadline)?;
            if !files.insert((pin.identity.device, pin.identity.inode)) {
                return Err(error(
                    LoadErrorKind::InvalidBundle,
                    "duplicate system library file",
                ));
            }
            libraries.push(pin);
        }
        let identity_sha256 = identity_digest(&libraries);
        check_deadline(deadline)?;
        let mut guard = state().lock().map_err(|_| poison())?;
        if let Some(failure) = &guard.failure {
            return Err(failure.clone());
        }
        if let Some(existing) = &guard.owner {
            if existing.identity_sha256 != identity_sha256 {
                return Err(error(
                    LoadErrorKind::DifferentBundle,
                    "system runtime changed within the process",
                ));
            }
            return Ok(Self {
                inner: Arc::clone(existing),
            });
        }
        // An already mapped exact file is not owned retrospectively by this
        // token. Preserve that distinction without blocking ordinary use.
        let identities: Vec<_> = libraries.iter().map(|pin| &pin.identity).collect();
        let preexisting = assess_preexisting_maps(linux::read_maps(), &identities)?;
        let preexisting_mapping_observed = preexisting.is_some();
        let preexisting_native_mappings = preexisting.unwrap_or_default();
        check_deadline(deadline)?;
        claim_native_mode(2)?;
        let inner = Arc::new(Inner {
            identity_sha256,
            libraries,
            preexisting_mapping_observed,
            preexisting_native_mappings,
        });
        guard.owner = Some(Arc::clone(&inner));
        Ok(Self { inner })
    }

    pub fn identity_sha256(&self) -> [u8; 32] {
        self.inner.identity_sha256
    }

    pub fn libraries(&self) -> Vec<SystemLibraryIdentity> {
        self.inner
            .libraries
            .iter()
            .map(|pin| pin.identity.clone())
            .collect()
    }

    /// Same-process `/proc/self/fd` path to the captured readonly descriptor.
    /// The static owner keeps the FD alive even if this token is dropped. This
    /// reduces replacement races for the explicitly loaded library only; it
    /// does not secure transitive SONAME lookup or native constructors.
    pub fn pinned_load_path(&self, role: &str) -> Result<PathBuf, LoadError> {
        let mut guard = state().lock().map_err(|_| poison())?;
        if let Some(failure) = &guard.failure {
            return Err(failure.clone());
        }
        let pin = self
            .inner
            .libraries
            .iter()
            .find(|pin| pin.identity.role == role)
            .ok_or_else(|| error(LoadErrorKind::InvalidBundle, "unknown system library role"))?;
        if let Err(failure) = check_still_named(pin) {
            guard.failure = Some(failure.clone());
            return Err(failure);
        }
        Ok(PathBuf::from(format!(
            "/proc/self/fd/{}",
            pin.file.as_raw_fd()
        )))
    }

    /// Observe current mappings after the caller's provider setup or forward.
    /// Strict verification is separate so that lazy libraries can be recorded.
    pub fn observe_mappings(&self) -> Result<SystemMappingObservation, LoadError> {
        self.with_latched_result(false)
    }

    /// Require every mandatory mapping and reject undeclared CUDA/ORT images.
    /// Failure is process-latched, including for already-cloned tokens.
    pub fn verify_mappings(&self) -> Result<SystemMappingObservation, LoadError> {
        self.with_latched_result(true)
    }

    fn with_latched_result(&self, strict: bool) -> Result<SystemMappingObservation, LoadError> {
        let mut guard = state().lock().map_err(|_| poison())?;
        if let Some(failure) = &guard.failure {
            return Err(failure.clone());
        }
        apply_observation(&mut guard, self.observe_inner(), strict)
    }

    fn observe_inner(&self) -> Result<SystemMappingObservation, LoadError> {
        for pin in &self.inner.libraries {
            check_still_named(pin)?;
        }
        let maps = match linux::read_maps() {
            Ok(maps) => maps,
            Err(_) => {
                return Ok(SystemMappingObservation {
                    identity_sha256: self.inner.identity_sha256,
                    availability: SystemMappingAvailability::Unavailable,
                    preexisting_mapping_observed: self.inner.preexisting_mapping_observed,
                    preexisting_native_mappings: self.inner.preexisting_native_mappings.clone(),
                    mapped_roles: Vec::new(),
                    missing_required_roles: Vec::new(),
                    missing_optional_roles: Vec::new(),
                    unregistered_native_mappings: Vec::new(),
                    gpu_provider_verified: false,
                    gpu_completion_verified: false,
                });
            }
        };
        let mut matched = BTreeSet::new();
        let mut unregistered = BTreeSet::new();
        let mut by_inode = BTreeMap::new();
        for pin in &self.inner.libraries {
            let (major, minor) = linux::device_parts(pin.identity.device);
            by_inode.insert((major, minor, pin.identity.inode), pin);
        }
        for mapping in maps {
            let key = (mapping.dev_major, mapping.dev_minor, mapping.inode);
            if let Some(pin) = by_inode.get(&key) {
                if mapping.deleted
                    || mapping.path.as_deref() != Some(pin.identity.canonical_path.as_path())
                {
                    return Err(error(
                        LoadErrorKind::MappingIdentity,
                        "mapped system library path or lifetime differs",
                    ));
                }
                matched.insert(pin.identity.role.clone());
            } else if let Some(name) = mapping
                .path
                .as_ref()
                .and_then(|p| p.file_name())
                .and_then(|s| s.to_str())
            {
                if linux::is_nvidia_runtime(name) || name.starts_with("libonnxruntime") {
                    // An additional version of a registered SONAME family is
                    // an identity conflict, not a harmless lazy dependency.
                    let family = native_soname_family(name);
                    if family.is_some_and(|family| {
                        self.inner.libraries.iter().any(|pin| {
                            pin.identity
                                .canonical_path
                                .file_name()
                                .and_then(|part| part.to_str())
                                .and_then(native_soname_family)
                                == Some(family)
                        })
                    }) {
                        return Err(error(
                            LoadErrorKind::MappingIdentity,
                            "a different image of a registered system library is mapped",
                        ));
                    }
                    unregistered.insert(name.to_owned());
                }
            }
        }
        let mut missing_required_roles = Vec::new();
        let mut missing_optional_roles = Vec::new();
        for pin in &self.inner.libraries {
            if !matched.contains(&pin.identity.role) {
                let missing = if pin.identity.required_mapping {
                    &mut missing_required_roles
                } else {
                    &mut missing_optional_roles
                };
                missing.push(pin.identity.role.clone());
            }
        }
        Ok(SystemMappingObservation {
            identity_sha256: self.inner.identity_sha256,
            availability: SystemMappingAvailability::Observed,
            preexisting_mapping_observed: self.inner.preexisting_mapping_observed,
            preexisting_native_mappings: self.inner.preexisting_native_mappings.clone(),
            mapped_roles: matched.into_iter().collect(),
            missing_required_roles,
            missing_optional_roles,
            unregistered_native_mappings: unregistered.into_iter().collect(),
            gpu_provider_verified: false,
            gpu_completion_verified: false,
        })
    }
}

fn apply_observation(
    state: &mut ProcessState,
    observation: Result<SystemMappingObservation, LoadError>,
    strict: bool,
) -> Result<SystemMappingObservation, LoadError> {
    if let Some(failure) = &state.failure {
        return Err(failure.clone());
    }
    let result =
        observation.and_then(|observation| assess_mapping_observation(observation, strict));
    if let Err(failure) = &result {
        // A transient unavailable observation must not poison a live
        // runtime; conflicts and identity failures remain latched.
        if !matches!(
            failure.kind,
            LoadErrorKind::ObservationUnavailable | LoadErrorKind::ForeignMapping
        ) {
            state.failure = Some(failure.clone());
        }
    }
    result
}

fn assess_preexisting_maps(
    maps: Result<Vec<linux::MapEntry>, LoadError>,
    registered: &[&SystemLibraryIdentity],
) -> Result<Option<Vec<String>>, LoadError> {
    let maps = match maps {
        Ok(maps) => maps,
        // No conclusion about pre-existing mappings can be drawn from an
        // unavailable or malformed process-map observation.
        Err(_) => return Ok(None),
    };
    let mut existing = BTreeSet::new();
    for mapping in maps {
        if let Some(name) = mapping
            .path
            .as_ref()
            .and_then(|p| p.file_name())
            .and_then(|s| s.to_str())
        {
            if linux::is_nvidia_runtime(name) || name.starts_with("libonnxruntime") {
                let identity = registered.iter().find(|identity| {
                    native_soname_family(name).is_some_and(|family| {
                        identity.canonical_path.file_name().and_then(|part| part.to_str())
                            .and_then(native_soname_family) == Some(family)
                    })
                });
                if let Some(identity) = identity {
                    let (major, minor) = linux::device_parts(identity.device);
                    if (mapping.dev_major, mapping.dev_minor, mapping.inode)
                        != (major, minor, identity.inode)
                        || mapping.deleted
                        || mapping.path.as_deref() != Some(identity.canonical_path.as_path())
                    {
                        return Err(error(
                            LoadErrorKind::MappingIdentity,
                            "preexisting registered system library identity differs",
                        ));
                    }
                }
                existing.insert(name.to_owned());
            }
        }
    }
    Ok(Some(existing.into_iter().collect()))
}

fn assess_mapping_observation(
    observation: SystemMappingObservation,
    strict: bool,
) -> Result<SystemMappingObservation, LoadError> {
    // Extra unrelated system dependencies are recorded for normal execution;
    // exact reproducibility requires an explicitly closed mapping inventory.
    if strict && !observation.unregistered_native_mappings.is_empty() {
        return Err(error(
            LoadErrorKind::ForeignMapping,
            "an unregistered system native library is mapped",
        ));
    }
    if strict && !observation.preexisting_native_mappings.is_empty() {
        return Err(error(
            LoadErrorKind::ForeignMapping,
            "native library ownership before capture cannot be established",
        ));
    }
    if strict
        && (observation.availability == SystemMappingAvailability::Unavailable
            || !observation.preexisting_mapping_observed)
    {
        return Err(error(
            LoadErrorKind::ObservationUnavailable,
            "system mapping evidence is unavailable",
        ));
    }
    if strict && !observation.missing_required_roles.is_empty() {
        return Err(error(
            LoadErrorKind::MissingMapping,
            "a required system library is not mapped",
        ));
    }
    Ok(observation)
}

fn native_soname_family(name: &str) -> Option<&str> {
    let (family, suffix) = name.split_once(".so")?;
    if family.is_empty() || !(suffix.is_empty() || suffix.starts_with('.')) {
        return None;
    }
    Some(family)
}

fn check_deadline(deadline: Instant) -> Result<(), LoadError> {
    if Instant::now() >= deadline {
        return Err(error(
            LoadErrorKind::ResourceLimit,
            "system capture deadline exceeded",
        ));
    }
    Ok(())
}

fn capture_one(
    request: SystemLibraryRequest,
    max_file_bytes: u64,
    remaining: &mut u64,
    deadline: Instant,
) -> Result<LibraryPin, LoadError> {
    let started = Instant::now();
    if !request.path.is_absolute() {
        return Err(error(
            LoadErrorKind::FileIdentity,
            "system library path must be absolute",
        ));
    }
    let path = fs::canonicalize(&request.path).map_err(|_| {
        error(
            LoadErrorKind::FileAccess,
            "system library cannot be resolved",
        )
    })?;
    if path.to_str().is_none_or(|p| p.contains(['\n', '\r', '\\'])) {
        return Err(error(
            LoadErrorKind::FileIdentity,
            "system library path cannot be represented in mapping evidence",
        ));
    }
    // Resolve normal SONAME symlinks, then prevent leaf replacement by a
    // symlink or a blocking FIFO between canonicalisation and open.
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(0x20000 | 0x800)
        .open(&path)
        .map_err(|_| error(LoadErrorKind::FileAccess, "system library cannot be opened"))?;
    let before = file.metadata().map_err(|_| {
        error(
            LoadErrorKind::FileAccess,
            "system library metadata is unavailable",
        )
    })?;
    if !before.is_file() {
        return Err(error(
            LoadErrorKind::FileIdentity,
            "system library is not a regular file",
        ));
    }
    if before.len() > max_file_bytes || before.len() > *remaining {
        return Err(error(
            LoadErrorKind::ResourceLimit,
            "system library byte limit exceeded",
        ));
    }
    *remaining -= before.len();
    check_deadline(deadline)?;
    let mut header = [0_u8; 20];
    file.read_exact(&mut header).map_err(|_| {
        error(
            LoadErrorKind::FileIdentity,
            "system ELF header is incomplete",
        )
    })?;
    let elf_class = header[4];
    let elf_machine = u16::from_le_bytes([header[18], header[19]]);
    let expected_machine = if cfg!(target_arch = "x86_64") {
        62
    } else if cfg!(target_arch = "aarch64") {
        183
    } else {
        0
    };
    if header[..4] != *b"\x7fELF"
        || elf_class != 2
        || header[5] != 1
        || header[6] != 1
        || u16::from_le_bytes([header[16], header[17]]) != 3
        || elf_machine != expected_machine
    {
        return Err(error(
            LoadErrorKind::FileIdentity,
            "system library ELF architecture or type differs",
        ));
    }
    let mut hash = Sha256::new();
    hash.update(header);
    let mut buffer = [0_u8; 64 * 1024];
    let mut count = header.len() as u64;
    loop {
        check_deadline(deadline)?;
        let n = file
            .read(&mut buffer)
            .map_err(|_| error(LoadErrorKind::FileAccess, "system library read failed"))?;
        if n == 0 {
            break;
        }
        count = count.checked_add(n as u64).ok_or_else(|| {
            error(
                LoadErrorKind::ResourceLimit,
                "system library length overflow",
            )
        })?;
        hash.update(&buffer[..n]);
    }
    check_deadline(deadline)?;
    let after = file.metadata().map_err(|_| {
        error(
            LoadErrorKind::FileAccess,
            "system library metadata is unavailable",
        )
    })?;
    let named = fs::metadata(&path).map_err(|_| {
        error(
            LoadErrorKind::FileAccess,
            "system library path is unavailable",
        )
    })?;
    if count != before.len() || !same_metadata(&before, &after) || !same_metadata(&before, &named) {
        return Err(error(
            LoadErrorKind::FileIdentity,
            "system library changed while being captured",
        ));
    }
    let identity = SystemLibraryIdentity {
        role: request.role,
        canonical_path: path,
        sha256: hash.finalize().into(),
        bytes: count,
        device: before.dev(),
        inode: before.ino(),
        elf_class,
        elf_machine,
        required_mapping: request.required_mapping,
        first_capture_elapsed_ns: started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
    };
    Ok(LibraryPin {
        file,
        metadata: before,
        identity,
    })
}

fn same_metadata(left: &Metadata, right: &Metadata) -> bool {
    left.dev() == right.dev()
        && left.ino() == right.ino()
        && left.len() == right.len()
        && left.mode() == right.mode()
        && left.mtime() == right.mtime()
        && left.mtime_nsec() == right.mtime_nsec()
        && left.ctime() == right.ctime()
        && left.ctime_nsec() == right.ctime_nsec()
}

fn check_still_named(pin: &LibraryPin) -> Result<(), LoadError> {
    let current = pin.file.metadata().map_err(|_| {
        error(
            LoadErrorKind::FileAccess,
            "system library descriptor is unavailable",
        )
    })?;
    let named = fs::metadata(&pin.identity.canonical_path).map_err(|_| {
        error(
            LoadErrorKind::FileAccess,
            "system library path is unavailable",
        )
    })?;
    if !same_metadata(&pin.metadata, &current) || !same_metadata(&pin.metadata, &named) {
        return Err(error(
            LoadErrorKind::FileIdentity,
            "captured system library was replaced or modified",
        ));
    }
    Ok(())
}

fn identity_digest(libraries: &[LibraryPin]) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"RoveZero/system-runtime-identity/v1\0");
    for pin in libraries {
        let identity = &pin.identity;
        let path = identity.canonical_path.to_string_lossy();
        digest.update((identity.role.len() as u64).to_le_bytes());
        digest.update(identity.role.as_bytes());
        digest.update((path.len() as u64).to_le_bytes());
        digest.update(path.as_bytes());
        digest.update(identity.sha256);
        digest.update(identity.bytes.to_le_bytes());
        digest.update(identity.device.to_le_bytes());
        digest.update(identity.inode.to_le_bytes());
        digest.update([identity.elf_class, u8::from(identity.required_mapping)]);
        digest.update(identity.elf_machine.to_le_bytes());
    }
    digest.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation(availability: SystemMappingAvailability) -> SystemMappingObservation {
        SystemMappingObservation {
            identity_sha256: [0; 32],
            availability,
            preexisting_mapping_observed: true,
            preexisting_native_mappings: Vec::new(),
            mapped_roles: vec!["ort-core".into()],
            missing_required_roles: Vec::new(),
            missing_optional_roles: Vec::new(),
            unregistered_native_mappings: Vec::new(),
            gpu_provider_verified: false,
            gpu_completion_verified: false,
        }
    }

    #[test]
    fn unavailable_maps_are_unknown_without_masking_observed_conflicts() {
        let unreadable = error(LoadErrorKind::FileAccess, "maps unavailable");
        assert_eq!(assess_preexisting_maps(Err(unreadable), &[]).unwrap(), None);
        assert_eq!(assess_preexisting_maps(Ok(Vec::new()), &[]).unwrap(), Some(vec![]));
        let foreign = linux::MapEntry {
            dev_major: 0,
            dev_minor: 0,
            inode: 1,
            path: Some("/usr/lib/libcudart.so.12".into()),
            deleted: false,
        };
        assert_eq!(
            assess_preexisting_maps(Ok(vec![foreign]), &[]).unwrap(),
            Some(vec!["libcudart.so.12".to_owned()])
        );

        let unknown = observation(SystemMappingAvailability::Unavailable);
        assert!(assess_mapping_observation(unknown.clone(), false).is_ok());
        assert_eq!(
            assess_mapping_observation(unknown, true).unwrap_err().kind,
            LoadErrorKind::ObservationUnavailable
        );
        let mut earlier_unknown = observation(SystemMappingAvailability::Observed);
        earlier_unknown.preexisting_mapping_observed = false;
        assert!(assess_mapping_observation(earlier_unknown.clone(), false).is_ok());
        assert_eq!(
            assess_mapping_observation(earlier_unknown, true)
                .unwrap_err()
                .kind,
            LoadErrorKind::ObservationUnavailable
        );
        let mut preexisting = observation(SystemMappingAvailability::Observed);
        preexisting.preexisting_native_mappings.push("libcudart.so.12".into());
        assert!(assess_mapping_observation(preexisting.clone(), false).is_ok());
        assert_eq!(
            assess_mapping_observation(preexisting, true).unwrap_err().kind,
            LoadErrorKind::ForeignMapping
        );
        let mut extra = observation(SystemMappingAvailability::Observed);
        extra
            .unregistered_native_mappings
            .push("libcudnn_ops.so.9".into());
        assert!(assess_mapping_observation(extra.clone(), false).is_ok());
        assert_eq!(
            assess_mapping_observation(extra, true)
                .unwrap_err()
                .kind,
            LoadErrorKind::ForeignMapping
        );
        assert_eq!(native_soname_family("libcudnn.so.9.2"), Some("libcudnn"));
        assert_ne!(native_soname_family("libcudnn.so.9"), native_soname_family("libcudnn_ops.so.9"));
    }

    #[test]
    fn preexisting_exact_image_is_observed_but_other_image_conflicts() {
        let identity = SystemLibraryIdentity {
            role: "cuda".into(),
            canonical_path: "/usr/lib/libcudart.so.12".into(),
            sha256: [0; 32],
            bytes: 1,
            device: 0,
            inode: 1,
            elf_class: 2,
            elf_machine: 62,
            required_mapping: true,
            first_capture_elapsed_ns: 0,
        };
        let mapping = |inode| linux::MapEntry {
            dev_major: 0,
            dev_minor: 0,
            inode,
            path: Some("/usr/lib/libcudart.so.12".into()),
            deleted: false,
        };
        assert_eq!(
            assess_preexisting_maps(Ok(vec![mapping(1)]), &[&identity]).unwrap(),
            Some(vec!["libcudart.so.12".to_owned()])
        );
        assert_eq!(
            assess_preexisting_maps(Ok(vec![mapping(2)]), &[&identity])
                .unwrap_err().kind,
            LoadErrorKind::MappingIdentity
        );
    }

    #[test]
    fn unavailable_strict_observation_does_not_poison_later_normal_run() {
        let mut state = ProcessState::default();
        let unknown = observation(SystemMappingAvailability::Unavailable);
        assert_eq!(
            apply_observation(&mut state, Ok(unknown), true)
                .unwrap_err()
                .kind,
            LoadErrorKind::ObservationUnavailable
        );
        assert!(state.failure.is_none());
        assert!(
            apply_observation(
                &mut state,
                Ok(observation(SystemMappingAvailability::Observed)),
                false
            )
            .is_ok()
        );
        let mut extra = observation(SystemMappingAvailability::Observed);
        extra
            .unregistered_native_mappings
            .push("libcudnn_ops.so.9".into());
        assert!(apply_observation(&mut state, Ok(extra.clone()), false).is_ok());
        assert_eq!(
            apply_observation(&mut state, Ok(extra), true)
                .unwrap_err()
                .kind,
            LoadErrorKind::ForeignMapping
        );
        assert!(state.failure.is_none());
        let conflict = error(LoadErrorKind::MappingIdentity, "conflicting library");
        assert_eq!(
            apply_observation(&mut state, Err(conflict), false)
            .unwrap_err()
            .kind,
            LoadErrorKind::MappingIdentity
        );
        assert_eq!(
            apply_observation(&mut state, Ok(observation(SystemMappingAvailability::Observed)), false)
                .unwrap_err().kind,
            LoadErrorKind::MappingIdentity
        );
    }

    #[test]
    fn capture_enforces_byte_and_elapsed_limits_before_hashing() {
        let path = std::env::current_exe().unwrap();
        let request = || SystemLibraryRequest {
            role: "test".to_owned(),
            path: path.clone(),
            required_mapping: false,
        };
        let mut remaining = 1;
        let error = capture_one(
            request(),
            1,
            &mut remaining,
            Instant::now() + Duration::from_secs(5),
        )
        .err()
        .unwrap();
        assert_eq!(error.kind, LoadErrorKind::ResourceLimit);

        let mut remaining = u64::MAX;
        let error = capture_one(request(), u64::MAX, &mut remaining, Instant::now())
            .err()
            .unwrap();
        assert_eq!(error.kind, LoadErrorKind::ResourceLimit);
    }
}
