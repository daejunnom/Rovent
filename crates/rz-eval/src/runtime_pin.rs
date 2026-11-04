//! A bootstrap-owned copy of the verified native library, outside the checkout.
//!
//! Hashing a caller's mutable path does not bind the bytes subsequently dlopened.
//! This capability copies verified bytes into a newly created private directory,
//! closes the writer, makes the copy read-only and keeps an OS file pin alive.
//! The caller must supply a private output root whose ancestors it controls.
//! Its owner must not chmod, replace or remove bootstrap copies. This is an
//! ownership contract, not protection against hostile code with the same OS UID.
//! Shared CPU content-addressed copies are verified again on every cache hit.
//! Runtime pins are retained for process lifetime once passed to ORT. No
//! runtime/provider assets are downloaded. Historical CUDA bundle declarations
//! remain readable, but bundle copying is disabled; system CUDA has its own
//! runtime and lifetime boundary.

use crate::{
    asset,
    error::{BackendError, FailureKind as K, FailureStage as S},
};
use serde::Deserialize;
use serde::Serialize;
use sha2::{Digest, Sha256};
#[cfg(target_os = "linux")]
use std::io::{Read, Seek, SeekFrom};
use std::{
    fs::{self, File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

const MAX_LIBRARY_BYTES: usize = 512 * 1024 * 1024;
const MAX_BUNDLE_JSON_BYTES: usize = 64 * 1024;
const MAX_BUNDLE_FILES: usize = 32;
const MAX_BUNDLE_FILE_BYTES: u64 = 1024 * 1024 * 1024;
const MAX_BUNDLE_BYTES: u64 = 4 * MAX_BUNDLE_FILE_BYTES;
const CORE_FILENAME: &str = "libonnxruntime.so.1.22.0";
const SHARED_FILENAME: &str = "libonnxruntime_providers_shared.so";
const CUDA_FILENAME: &str = "libonnxruntime_providers_cuda.so";
const NVIDIA_FILENAMES: &[&str] = &[
    "libcudart.so.12",
    "libcublas.so.12",
    "libcublasLt.so.12",
    "libcufft.so.11",
    "libcurand.so.10",
    "libnvrtc.so.12",
    "libnvrtc-builtins.so.12.8",
    "libnvJitLink.so.12",
    "libcudnn.so.9",
    "libcudnn_adv.so.9",
    "libcudnn_cnn.so.9",
    "libcudnn_ops.so.9",
    "libcudnn_engines_precompiled.so.9",
    "libcudnn_engines_runtime_compiled.so.9",
    "libcudnn_graph.so.9",
    "libcudnn_heuristic.so.9",
];
static NEXT_COPY: AtomicU64 = AtomicU64::new(0);

mod cache;
pub use cache::RuntimeCache;
mod aliases;
pub use aliases::RuntimeBundleAlias;
mod artifact_tree;
pub use artifact_tree::{PreparedCudaArtifactTree, PreparedCudaArtifactTreeObservation};

/// Storage provenance is distinct from model-evaluation cache provenance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeStorageOrigin {
    PrivateCopy,
    CacheCreated,
    CacheReused,
}

/// No private paths. Reuse still requires byte verification and live file pins.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct RuntimeStorage {
    pub origin: RuntimeStorageOrigin,
    pub cache_key: Option<[u8; 32]>,
    pub bytes: u64,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeBundleFileRole {
    Core,
    ProvidersShared,
    ProvidersCuda,
    NvidiaDependency,
}

#[derive(Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RuntimeBundleFile {
    pub role: RuntimeBundleFileRole,
    pub filename: String,
    pub bytes: u64,
    pub sha256: String,
}

impl std::fmt::Debug for RuntimeBundleFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let filename = if allowed_bundle_filename(self.role, &self.filename) {
            self.filename.as_str()
        } else {
            "<invalid library basename>"
        };
        f.debug_struct("RuntimeBundleFile")
            .field("role", &self.role)
            .field("filename", &filename)
            .field("bytes", &self.bytes)
            .field("sha256", &"<redacted>")
            .finish()
    }
}

/// The first Linux GPU profile requires ORT 1.22 plus all 16 declared NVIDIA
/// libraries. This is a file identity manifest; ELF resolution is a separate gate.
/// Only deserialization is provided: shared diagnostics use the safe Debug view.
#[derive(Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CudaRuntimeBundleSpec {
    pub schema_version: u32,
    pub files: Vec<RuntimeBundleFile>,
    /// V2 only: a closed, explicit set of relative cuDNN loader basenames.
    /// Omitted in V1; it never changes that format's canonical digest.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loader_aliases: Option<Vec<RuntimeBundleAlias>>,
}

impl std::fmt::Debug for CudaRuntimeBundleSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaRuntimeBundleSpec")
            .field("schema_version", &self.schema_version)
            .field("files", &self.files)
            .field("loader_aliases", &self.loader_aliases)
            .finish()
    }
}

impl CudaRuntimeBundleSpec {
    pub fn from_json(json: &str) -> Result<Self, BackendError> {
        if json.len() > MAX_BUNDLE_JSON_BYTES {
            return Err(bundle_limit_error("CUDA bundle JSON exceeds 64 KiB"));
        }
        // Derived structs reject duplicate fields at both manifest/file levels.
        // Value-based parsing would lose that information and is not used here.
        let spec: Self = serde_json::from_str(json).map_err(|error| {
            bundle_error("CUDA bundle JSON is malformed")
                .with_external_cause(crate::error::CauseCode::RuntimePath, &error)
        })?;
        spec.validate()?;
        Ok(spec)
    }

    pub fn validate(&self) -> Result<(), BackendError> {
        if !matches!(self.schema_version, 1 | 2) {
            return Err(bundle_error("CUDA bundle schema version is unsupported"));
        }
        if self.files.len() > MAX_BUNDLE_FILES {
            return Err(bundle_limit_error("CUDA bundle exceeds 32 files"));
        }
        let mut required = [false; 3];
        let mut total = 0_u64;
        for (index, file) in self.files.iter().enumerate() {
            if !allowed_bundle_filename(file.role, &file.filename) {
                return Err(bundle_error("CUDA bundle filename or role is unsupported"));
            }
            if self.files[..index]
                .iter()
                .any(|previous| previous.filename == file.filename)
            {
                return Err(bundle_error("CUDA bundle contains a duplicate filename"));
            }
            if file.bytes == 0 || file.bytes > MAX_BUNDLE_FILE_BYTES {
                return Err(bundle_limit_error(
                    "CUDA bundle file size is outside its budget",
                ));
            }
            total = total
                .checked_add(file.bytes)
                .ok_or_else(|| bundle_limit_error("CUDA bundle total byte count overflowed"))?;
            if total > MAX_BUNDLE_BYTES {
                return Err(bundle_limit_error("CUDA bundle exceeds 4 GiB"));
            }
            bundle_file_digest(file)?;
            match file.role {
                RuntimeBundleFileRole::Core => required[0] = true,
                RuntimeBundleFileRole::ProvidersShared => required[1] = true,
                RuntimeBundleFileRole::ProvidersCuda => required[2] = true,
                RuntimeBundleFileRole::NvidiaDependency => {}
            }
        }
        if required != [true; 3] {
            return Err(bundle_error("CUDA bundle lacks required ORT libraries"));
        }
        if self.files.len() != 3 + NVIDIA_FILENAMES.len()
            || NVIDIA_FILENAMES.iter().any(|filename| {
                !self.files.iter().any(|file| {
                    file.role == RuntimeBundleFileRole::NvidiaDependency
                        && file.filename == *filename
                })
            })
        {
            return Err(bundle_error(
                "CUDA bundle lacks first-profile NVIDIA dependencies",
            ));
        }
        aliases::validate(self)?;
        Ok(())
    }

    /// SHA-256 of a domain-separated binary codec, independent of JSON order.
    /// Codec: domain, u32-BE schema/count, then filename-sorted entries containing
    /// u32-BE name length, name bytes, u8 role, u64-BE size and 32 digest bytes.
    pub fn digest(&self) -> Result<[u8; 32], BackendError> {
        self.validate()?;
        let mut files: Vec<_> = self.files.iter().collect();
        files.sort_unstable_by(|left, right| left.filename.cmp(&right.filename));
        let mut digest = Sha256::new();
        digest.update(match self.schema_version {
            1 => b"RoveZero/native-cuda-bundle\0binary-codec-v1\0".as_slice(),
            2 => b"RoveZero/native-cuda-bundle\0binary-codec-v2-aliases\0".as_slice(),
            _ => unreachable!("validated schema"),
        });
        digest.update(self.schema_version.to_be_bytes());
        digest.update((files.len() as u32).to_be_bytes());
        for file in files {
            digest.update((file.filename.len() as u32).to_be_bytes());
            digest.update(file.filename.as_bytes());
            digest.update([match file.role {
                RuntimeBundleFileRole::Core => 1,
                RuntimeBundleFileRole::ProvidersShared => 2,
                RuntimeBundleFileRole::ProvidersCuda => 3,
                RuntimeBundleFileRole::NvidiaDependency => 4,
            }]);
            digest.update(file.bytes.to_be_bytes());
            digest.update(bundle_file_digest(file)?);
        }
        if let Some(aliases) = &self.loader_aliases {
            let mut aliases: Vec<_> = aliases.iter().collect();
            aliases.sort_unstable_by(|left, right| left.filename.cmp(&right.filename));
            digest.update((aliases.len() as u32).to_be_bytes());
            for alias in aliases {
                for basename in [&alias.filename, &alias.target] {
                    digest.update((basename.len() as u32).to_be_bytes());
                    digest.update(basename.as_bytes());
                }
            }
        }
        Ok(digest.finalize().into())
    }
}

fn allowed_bundle_filename(role: RuntimeBundleFileRole, filename: &str) -> bool {
    match role {
        RuntimeBundleFileRole::Core => filename == CORE_FILENAME,
        RuntimeBundleFileRole::ProvidersShared => filename == SHARED_FILENAME,
        RuntimeBundleFileRole::ProvidersCuda => filename == CUDA_FILENAME,
        RuntimeBundleFileRole::NvidiaDependency => NVIDIA_FILENAMES.contains(&filename),
    }
}

fn bundle_file_digest(file: &RuntimeBundleFile) -> Result<[u8; 32], BackendError> {
    if file.sha256.len() != 64
        || !file
            .sha256
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(bundle_error("CUDA bundle SHA-256 must be lowercase hex"));
    }
    asset::parse_sha256(&file.sha256)
}

fn bundle_error(detail: &'static str) -> BackendError {
    BackendError::new(K::UnsupportedModel, S::Backend, detail)
}

fn bundle_limit_error(detail: &'static str) -> BackendError {
    BackendError::new(K::ResourceExhausted, S::Backend, detail)
}

struct OwnedCudaBundle {
    digest: [u8; 32],
    schema_version: u32,
    files: Vec<RuntimeBundleFile>,
    loader_aliases: Option<Vec<RuntimeBundleAlias>>,
    // The core handle is OwnedCopy::_pin; every other bundle file is held here.
    _pins: Vec<File>,
}

/// Crate-private handoff to the audited native loader. The path must never be
/// printed as a diagnostic. File clones have independent lifetimes but share
/// their seek offset; consumers must not perform concurrent seek/stream reads.
/// Positional reads through read_at/pread do not share that offset.
#[cfg_attr(not(all(feature = "onnx", target_os = "linux")), allow(dead_code))]
pub(crate) struct PinnedBundleLibrary {
    pub path: PathBuf,
    pub file: File,
    pub digest: [u8; 32],
    pub bytes: u64,
}

struct OwnedCopy {
    path: PathBuf,
    digest: [u8; 32],
    // Windows denies write/delete sharing; Unix retains the exact owned inode.
    _pin: File,
    bundle: Option<OwnedCudaBundle>,
    storage: RuntimeStorage,
}

/// Capability issued only after an owned copy has been verified and pinned.
#[derive(Clone)]
pub struct RuntimeLibraryPin(Arc<OwnedCopy>);

impl std::fmt::Debug for RuntimeLibraryPin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Private filesystem paths are not shared diagnostics.
        f.debug_struct("RuntimeLibraryPin")
            .field("cuda_bundle", &self.0.bundle.is_some())
            .finish_non_exhaustive()
    }
}

impl RuntimeLibraryPin {
    /// `output_root` must exist outside Git and be owned by this bootstrap.
    /// Source mutation after this returns cannot change the loaded copy.
    pub fn copy_verified(
        source: &Path,
        output_root: &Path,
        expected_sha256: &str,
    ) -> Result<Self, BackendError> {
        Self::copy_with(source, output_root, expected_sha256, |writer, bytes| {
            writer.write_all(bytes)?;
            writer.sync_all()
        })
    }

    fn copy_with(
        source: &Path,
        output_root: &Path,
        expected_sha256: &str,
        write: impl FnOnce(&mut File, &[u8]) -> std::io::Result<()>,
    ) -> Result<Self, BackendError> {
        let expected = asset::parse_sha256(expected_sha256)?;
        let root_metadata = fs::symlink_metadata(output_root)
            .map_err(|_| io_error("runtime output root is unavailable"))?;
        if !root_metadata.is_dir() || root_metadata.file_type().is_symlink() {
            return Err(io_error(
                "runtime output root must be a real owned directory",
            ));
        }
        let root = output_root
            .canonicalize()
            .map_err(|_| io_error("cannot resolve runtime output root"))?;
        let bytes = asset::read_bounded(source, MAX_LIBRARY_BYTES)?;
        if bytes.is_empty() || asset::sha256(&bytes) != expected {
            return Err(BackendError::new(
                K::IdentityMismatch,
                S::Backend,
                "native runtime source digest differs",
            ));
        }
        let filename = source
            .file_name()
            .ok_or_else(|| io_error("runtime source has no library filename"))?;
        let sequence = NEXT_COPY
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_add(1))
            .map_err(|_| {
                BackendError::new(
                    K::ResourceExhausted,
                    S::Backend,
                    "runtime copy sequence exhausted",
                )
            })?;
        let directory = root.join(format!("ort-bootstrap-{}-{}", std::process::id(), sequence));
        let builder = fs::DirBuilder::new();
        #[cfg(unix)]
        let builder = {
            use std::os::unix::fs::DirBuilderExt;
            let mut builder = builder;
            builder.mode(0o700);
            builder
        };
        // create, never create_dir_all: a collision cannot be reused.
        builder
            .create(&directory)
            .map_err(|_| io_error("cannot create exclusive runtime directory"))?;
        let path = directory.join(filename);
        let result = (|| {
            let mut options = OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            let mut writer = options
                .open(&path)
                .map_err(|_| io_error("cannot create exclusive runtime copy"))?;
            write(&mut writer, &bytes).map_err(|error| {
                io_error("cannot write and sync verified runtime copy")
                    .with_external_cause(crate::error::CauseCode::RuntimePath, &error)
            })?;
            drop(writer);
            let mut permissions = fs::metadata(&path)
                .map_err(|_| io_error("runtime copy metadata failed"))?
                .permissions();
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                permissions.set_mode(0o400);
            }
            #[cfg(not(unix))]
            permissions.set_readonly(true);
            fs::set_permissions(&path, permissions)
                .map_err(|_| io_error("cannot make runtime copy read-only"))?;
            let mut options = OpenOptions::new();
            options.read(true);
            #[cfg(windows)]
            {
                use std::os::windows::fs::OpenOptionsExt;
                // FILE_SHARE_READ: denies all writers and delete/rename handles.
                options.share_mode(1);
            }
            let pin = options
                .open(&path)
                .map_err(|_| io_error("cannot pin verified runtime copy"))?;
            // Re-read after closing the writer and acquiring the durable pin.
            if asset::sha256(&asset::read_bounded(&path, MAX_LIBRARY_BYTES)?) != expected {
                return Err(BackendError::new(
                    K::IdentityMismatch,
                    S::Backend,
                    "runtime owned copy digest differs",
                ));
            }
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(&directory, fs::Permissions::from_mode(0o500))
                    .map_err(|_| io_error("cannot seal runtime directory"))?;
            }
            Ok(Self(Arc::new(OwnedCopy {
                path: path.clone(),
                digest: expected,
                _pin: pin,
                bundle: None,
                storage: RuntimeStorage {
                    origin: RuntimeStorageOrigin::PrivateCopy,
                    cache_key: None,
                    bytes: bytes.len() as u64,
                },
            })))
        })();
        // Only a failed, unpublished copy is eligible for local cleanup.
        if result.is_err() {
            #[cfg(windows)]
            let _ = clear_owned_windows_readonly_attribute(&path);
            let _ = fs::remove_file(&path);
            let _ = fs::remove_dir(&directory);
        }
        result
    }

    pub fn binary_digest(&self) -> [u8; 32] {
        self.0.digest
    }

    pub fn storage(&self) -> RuntimeStorage {
        self.0.storage
    }

    pub fn bundle_digest(&self) -> Option<[u8; 32]> {
        self.0.bundle.as_ref().map(|bundle| bundle.digest)
    }

    pub fn bundle_files(&self) -> Option<&[RuntimeBundleFile]> {
        self.0.bundle.as_ref().map(|bundle| bundle.files.as_slice())
    }

    /// Exact registered manifest semantics, including V2 loader aliases.
    /// Consumers must not reconstruct a V1 manifest from `bundle_files` alone.
    pub fn bundle_spec(&self) -> Option<CudaRuntimeBundleSpec> {
        self.0.bundle.as_ref().map(|bundle| CudaRuntimeBundleSpec {
            schema_version: bundle.schema_version,
            files: bundle.files.clone(),
            loader_aliases: bundle.loader_aliases.clone(),
        })
    }

    #[cfg_attr(not(all(feature = "onnx", target_os = "linux")), allow(dead_code))]
    pub(crate) fn try_clone_nvidia_pins(&self) -> Result<Vec<PinnedBundleLibrary>, BackendError> {
        self.try_clone_bundle_pins(false)
    }

    #[cfg_attr(not(all(feature = "onnx", target_os = "linux")), allow(dead_code))]
    pub(crate) fn try_clone_ort_pins(&self) -> Result<Vec<PinnedBundleLibrary>, BackendError> {
        self.try_clone_bundle_pins(true)
    }

    #[cfg_attr(not(all(feature = "onnx", target_os = "linux")), allow(dead_code))]
    fn try_clone_bundle_pins(&self, ort: bool) -> Result<Vec<PinnedBundleLibrary>, BackendError> {
        let bundle = self.0.bundle.as_ref().ok_or_else(|| {
            BackendError::new(
                K::BackendUnavailable,
                S::Backend,
                "CPU runtime pin has no CUDA bundle",
            )
        })?;
        #[cfg(target_os = "linux")]
        {
            use std::os::unix::fs::{MetadataExt, PermissionsExt};
            let directory = self
                .0
                .path
                .parent()
                .ok_or_else(|| bundle_error("CUDA bundle pin has no owned directory"))?;
            if bundle._pins.len() + 1 != bundle.files.len() {
                return Err(bundle_error("CUDA bundle handle count differs"));
            }
            aliases::verify(
                directory,
                bundle.loader_aliases.as_deref().unwrap_or(&[]),
                &bundle.files,
            )?;
            let mut result = Vec::with_capacity(if ort { 3 } else { NVIDIA_FILENAMES.len() });
            let mut other_pins = bundle._pins.iter();
            for entry in &bundle.files {
                let pin = if entry.role == RuntimeBundleFileRole::Core {
                    &self.0._pin
                } else {
                    other_pins
                        .next()
                        .ok_or_else(|| bundle_error("CUDA bundle retained handle is missing"))?
                };
                if (entry.role != RuntimeBundleFileRole::NvidiaDependency) != ort {
                    continue;
                }
                let path = directory.join(&entry.filename);
                let retained = pin.metadata().map_err(|error| {
                    bundle_io_error("CUDA retained pin metadata failed", &error)
                })?;
                let named = fs::symlink_metadata(&path).map_err(|error| {
                    bundle_io_error("CUDA owned pin path metadata failed", &error)
                })?;
                if !retained.is_file()
                    || !named.is_file()
                    || named.file_type().is_symlink()
                    || retained.dev() != named.dev()
                    || retained.ino() != named.ino()
                    || retained.len() != entry.bytes
                    || named.len() != entry.bytes
                    || retained.nlink() != 1
                    || named.nlink() != 1
                    || retained.permissions().mode() & 0o777 != 0o400
                    || named.permissions().mode() & 0o777 != 0o400
                {
                    return Err(bundle_identity_error(
                        "CUDA owned pin inode or permissions differ",
                    ));
                }
                let file = pin.try_clone().map_err(|error| {
                    bundle_io_error("cannot clone CUDA retained file pin", &error)
                })?;
                result.push(PinnedBundleLibrary {
                    path,
                    file,
                    digest: bundle_file_digest(entry)?,
                    bytes: entry.bytes,
                });
            }
            Ok(result)
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (bundle, ort);
            Err(BackendError::new(
                K::BackendUnavailable,
                S::Backend,
                "CUDA runtime bundle pins currently require Linux",
            ))
        }
    }

    /// Historical CUDA bundle metadata remains readable, but new bundle
    /// copies are disabled. Use the registered system CUDA runtime instead.
    pub fn copy_cuda_bundle(
        source_root: &Path,
        output_root: &Path,
        spec: &CudaRuntimeBundleSpec,
    ) -> Result<Self, BackendError> {
        refuse_legacy_cuda_bundle()?;
        spec.validate()?;
        #[cfg(target_os = "linux")]
        {
            Self::copy_linux_cuda_bundle(source_root, output_root, spec)
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (source_root, output_root);
            Err(BackendError::new(
                K::BackendUnavailable,
                S::Backend,
                "CUDA runtime bundle pins currently require Linux",
            ))
        }
    }

    #[cfg(target_os = "linux")]
    fn copy_linux_cuda_bundle(
        source_root: &Path,
        output_root: &Path,
        spec: &CudaRuntimeBundleSpec,
    ) -> Result<Self, BackendError> {
        Self::copy_linux_cuda_bundle_observed(source_root, output_root, spec, &mut |_| Ok(()))
    }

    #[cfg(target_os = "linux")]
    fn copy_linux_cuda_bundle_observed(
        source_root: &Path,
        output_root: &Path,
        spec: &CudaRuntimeBundleSpec,
        checkpoint: &mut dyn FnMut(u64) -> Result<(), BackendError>,
    ) -> Result<Self, BackendError> {
        use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt};
        checkpoint(0)?;
        let source_root = real_bundle_root(source_root)?;
        let output_root = real_bundle_root(output_root)?;
        let bundle_digest = spec.digest()?;
        let sequence = NEXT_COPY
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_add(1))
            .map_err(|_| bundle_limit_error("runtime copy sequence exhausted"))?;
        let directory =
            output_root.join(format!("ort-bootstrap-{}-{}", std::process::id(), sequence));
        fs::DirBuilder::new()
            .mode(0o700)
            .create(&directory)
            .map_err(|error| {
                bundle_io_error("cannot create exclusive CUDA bundle directory", &error)
            })?;
        let mut created_files =
            Vec::with_capacity(spec.files.len() + spec.loader_aliases.as_ref().map_or(0, Vec::len));
        let result = (|| {
            let mut files = spec.files.clone();
            files.sort_unstable_by(|left, right| left.filename.cmp(&right.filename));
            let mut core = None;
            let mut pins = Vec::with_capacity(files.len() - 1);
            for file in &files {
                checkpoint(0)?;
                let source = source_root.join(&file.filename);
                let metadata = fs::symlink_metadata(&source).map_err(|error| {
                    bundle_io_error("CUDA bundle source metadata failed", &error)
                })?;
                if !metadata.is_file() || metadata.file_type().is_symlink() || metadata.nlink() != 1
                {
                    return Err(bundle_error(
                        "CUDA bundle source must be a single-link regular nonsymlink file",
                    ));
                }
                // Linux O_NOFOLLOW closes the symlink swap window; O_NONBLOCK
                // avoids waiting on a FIFO swapped in before the handle check.
                let mut reader = OpenOptions::new()
                    .read(true)
                    .custom_flags(0x20000 | 0x800)
                    .open(&source)
                    .map_err(|error| bundle_io_error("cannot open CUDA bundle source", &error))?;
                verify_bundle_file_length(&reader, file.bytes)?;
                if reader
                    .metadata()
                    .map_err(|error| {
                        bundle_io_error("CUDA bundle source pin metadata failed", &error)
                    })?
                    .nlink()
                    != 1
                {
                    return Err(bundle_identity_error(
                        "CUDA bundle source pin is hard-linked",
                    ));
                }
                let path = directory.join(&file.filename);
                let mut writer = OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .mode(0o600)
                    .open(&path)
                    .map_err(|error| {
                        bundle_io_error("cannot create exclusive CUDA bundle file", &error)
                    })?;
                created_files.push(path.clone());
                let expected = bundle_file_digest(file)?;
                if stream_bundle_file_observed(
                    &mut reader,
                    Some(&mut writer),
                    file.bytes,
                    checkpoint,
                )? != expected
                {
                    return Err(bundle_identity_error("CUDA bundle source digest differs"));
                }
                writer
                    .sync_all()
                    .map_err(|error| bundle_io_error("cannot sync CUDA bundle file", &error))?;
                drop(writer);
                fs::set_permissions(&path, fs::Permissions::from_mode(0o400)).map_err(|error| {
                    bundle_io_error("cannot make CUDA bundle file read-only", &error)
                })?;
                let mut pin = File::open(&path)
                    .map_err(|error| bundle_io_error("cannot pin CUDA bundle file", &error))?;
                verify_bundle_file_length(&pin, file.bytes)?;
                if stream_bundle_file_observed(&mut pin, None, file.bytes, checkpoint)? != expected
                {
                    return Err(bundle_identity_error(
                        "CUDA bundle owned copy digest differs",
                    ));
                }
                pin.seek(SeekFrom::Start(0)).map_err(|error| {
                    bundle_io_error("cannot rewind CUDA bundle file pin", &error)
                })?;
                if file.role == RuntimeBundleFileRole::Core {
                    core = Some((path, expected, pin));
                } else {
                    pins.push(pin);
                }
            }
            for alias in spec.loader_aliases.as_deref().unwrap_or(&[]) {
                checkpoint(0)?;
                let path = directory.join(&alias.filename);
                std::os::unix::fs::symlink(&alias.target, &path).map_err(|error| {
                    bundle_io_error("cannot create declared CUDA loader alias", &error)
                })?;
                created_files.push(path);
            }
            aliases::verify(
                &directory,
                spec.loader_aliases.as_deref().unwrap_or(&[]),
                &files,
            )?;
            let (path, digest, pin) =
                core.ok_or_else(|| bundle_error("CUDA bundle core pin is missing"))?;
            fs::set_permissions(&directory, fs::Permissions::from_mode(0o500))
                .map_err(|error| bundle_io_error("cannot seal CUDA bundle directory", &error))?;
            Ok(Self(Arc::new(OwnedCopy {
                path,
                digest,
                _pin: pin,
                bundle: Some(OwnedCudaBundle {
                    digest: bundle_digest,
                    schema_version: spec.schema_version,
                    files,
                    loader_aliases: spec.loader_aliases.clone(),
                    _pins: pins,
                }),
                storage: RuntimeStorage {
                    origin: RuntimeStorageOrigin::PrivateCopy,
                    cache_key: None,
                    bytes: spec.files.iter().map(|file| file.bytes).sum(),
                },
            })))
        })();
        if result.is_err() {
            // Every handle in the failed closure has closed. Only paths created
            // with create_new in our exclusive unpublished directory are removed.
            // Cleanup is best-effort: IO failure may leave an unpublished copy
            // for caller-owned cleanup. The original failure remains authoritative.
            let _ = fs::set_permissions(&directory, fs::Permissions::from_mode(0o700));
            for path in created_files.iter().rev() {
                let _ = fs::remove_file(path);
            }
            let _ = fs::remove_dir(&directory);
        }
        result
    }

    #[cfg_attr(not(feature = "onnx"), allow(dead_code))]
    pub(crate) fn path(&self) -> &Path {
        &self.0.path
    }
}

fn refuse_legacy_cuda_bundle() -> Result<(), BackendError> {
    Err(BackendError::new(
        K::BackendUnavailable,
        S::Backend,
        "legacy CUDA library bundle execution is disabled; use a system CUDA runtime",
    ))
}

#[cfg(target_os = "linux")]
fn real_bundle_root(root: &Path) -> Result<PathBuf, BackendError> {
    let metadata = fs::symlink_metadata(root)
        .map_err(|error| bundle_io_error("CUDA bundle root is unavailable", &error))?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err(bundle_error(
            "CUDA bundle root must be a real owned directory",
        ));
    }
    root.canonicalize()
        .map_err(|error| bundle_io_error("cannot resolve CUDA bundle root", &error))
}

#[cfg(target_os = "linux")]
fn verify_bundle_file_length(file: &File, expected: u64) -> Result<(), BackendError> {
    let metadata = file
        .metadata()
        .map_err(|error| bundle_io_error("CUDA bundle opened file metadata failed", &error))?;
    if !metadata.is_file() || metadata.len() != expected {
        return Err(bundle_identity_error("CUDA bundle file size differs"));
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn stream_bundle_file_observed(
    reader: &mut File,
    mut writer: Option<&mut File>,
    expected_bytes: u64,
    checkpoint: &mut dyn FnMut(u64) -> Result<(), BackendError>,
) -> Result<[u8; 32], BackendError> {
    let mut buffer = [0_u8; 64 * 1024];
    let mut remaining = expected_bytes;
    let mut digest = Sha256::new();
    while remaining > 0 {
        let capacity = remaining.min(buffer.len() as u64) as usize;
        checkpoint(capacity as u64)?;
        let count = reader
            .read(&mut buffer[..capacity])
            .map_err(|error| bundle_io_error("cannot read CUDA bundle bytes", &error))?;
        if count == 0 {
            return Err(bundle_identity_error(
                "CUDA bundle bytes ended before declared size",
            ));
        }
        if let Some(writer) = writer.as_mut() {
            writer
                .write_all(&buffer[..count])
                .map_err(|error| bundle_io_error("cannot write CUDA bundle bytes", &error))?;
        }
        digest.update(&buffer[..count]);
        remaining -= count as u64;
    }
    // Reserve the bounded EOF probe too; short reads remain conservatively
    // charged and a failing owner never refunds an uncertain read.
    checkpoint(1)?;
    if reader
        .read(&mut buffer[..1])
        .map_err(|error| bundle_io_error("cannot verify CUDA bundle end", &error))?
        != 0
    {
        return Err(bundle_identity_error(
            "CUDA bundle bytes exceed declared size",
        ));
    }
    Ok(digest.finalize().into())
}

#[cfg(target_os = "linux")]
fn bundle_identity_error(detail: &'static str) -> BackendError {
    BackendError::new(K::IdentityMismatch, S::Backend, detail)
}

#[cfg(target_os = "linux")]
fn bundle_io_error(detail: &'static str, error: &std::io::Error) -> BackendError {
    io_error(detail).with_external_cause(crate::error::CauseCode::RuntimePath, error)
}

fn io_error(detail: &'static str) -> BackendError {
    BackendError::new(K::Io, S::Backend, detail)
}

// Windows only: clear FILE_ATTRIBUTE_READONLY on this bootstrap's own copy.
// Unlike Unix mode changes, this does not grant world write permission. Cleanup
// calls this after its writer/pin has closed; the sharing regression also clears
// the attribute while pinned to verify that write/delete sharing stays denied.
#[cfg(windows)]
#[allow(clippy::permissions_set_readonly_false)]
fn clear_owned_windows_readonly_attribute(path: &Path) -> std::io::Result<()> {
    let mut permissions = fs::metadata(path)?.permissions();
    permissions.set_readonly(false);
    fs::set_permissions(path, permissions)
}

#[cfg(test)]
mod tests {
    use super::*;
    static NEXT_TEST: AtomicU64 = AtomicU64::new(0);
    const FAKE_LIBRARY: &[u8] = b"fake native bootstrap bytes: no actual ORT execution";

    fn fixture_root() -> (PathBuf, PathBuf) {
        #[cfg(windows)]
        let base =
            PathBuf::from(std::env::var_os("APPDATA").expect("Windows tests require APPDATA"));
        #[cfg(not(windows))]
        let base = std::env::var_os("RUNNER_TEMP")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        let base = base.join("RoveZero").join("tmp").join("native-pin-tests");
        fs::create_dir_all(&base).unwrap();
        let base = base.canonicalize().unwrap();
        let root = base.join(format!(
            "{}-{}",
            std::process::id(),
            NEXT_TEST.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        let source = root.join("library.fixture");
        fs::write(&source, FAKE_LIBRARY).unwrap();
        (root, source)
    }
    fn release_copy(pin: RuntimeLibraryPin, root: &Path) {
        let path = pin.path().to_owned();
        let directory = path.parent().unwrap().to_owned();
        assert_eq!(directory.parent().unwrap(), root);
        drop(pin);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&directory, fs::Permissions::from_mode(0o700)).unwrap();
        }
        #[cfg(windows)]
        clear_owned_windows_readonly_attribute(&path).unwrap();
        fs::remove_file(path).unwrap();
        fs::remove_dir(directory).unwrap();
    }
    fn release_root(root: PathBuf, source: PathBuf) {
        assert_eq!(source.parent().unwrap(), root);
        fs::remove_file(source).unwrap();
        fs::remove_dir(root).unwrap();
    }

    #[test]
    fn mutable_source_does_not_change_owned_bytes_or_disclose_private_path() {
        let (root, source) = fixture_root();
        let digest = asset::sha256(FAKE_LIBRARY);
        let pin =
            RuntimeLibraryPin::copy_verified(&source, &root, &asset::hex_sha256(FAKE_LIBRARY))
                .unwrap();
        fs::write(&source, b"source was subsequently replaced").unwrap();
        assert_eq!(asset::sha256(&fs::read(pin.path()).unwrap()), digest);
        assert_eq!(pin.binary_digest(), digest);
        let diagnostic = format!("{pin:?}");
        assert!(!diagnostic.contains(root.to_str().unwrap()));
        assert!(!diagnostic.contains("library.fixture"));
        release_copy(pin, &root);
        release_root(root, source);
    }

    #[test]
    fn digest_rejection_and_partial_write_failure_leave_no_copy() {
        let (root, source) = fixture_root();
        let failure =
            RuntimeLibraryPin::copy_verified(&source, &root, &"01".repeat(32)).unwrap_err();
        assert_eq!(failure.kind, K::IdentityMismatch);
        assert_eq!(fs::read_dir(&root).unwrap().count(), 1);
        let failure = RuntimeLibraryPin::copy_with(
            &source,
            &root,
            &asset::hex_sha256(FAKE_LIBRARY),
            |writer, bytes| {
                writer.write_all(&bytes[..8])?;
                Err(std::io::Error::other(
                    "injected finite partial-write failure",
                ))
            },
        )
        .unwrap_err();
        assert_eq!(failure.kind, K::Io);
        assert!(failure.cause.is_some());
        assert_eq!(
            fs::read_dir(&root).unwrap().count(),
            1,
            "only the caller's source remains after failure cleanup"
        );
        release_root(root, source);
    }

    #[cfg(windows)]
    #[test]
    fn live_pin_denies_write_and_delete_sharing() {
        use std::os::windows::fs::OpenOptionsExt;
        let (root, source) = fixture_root();
        let pin =
            RuntimeLibraryPin::copy_verified(&source, &root, &asset::hex_sha256(FAKE_LIBRARY))
                .unwrap();
        // Clear readonly to test OS sharing rather than just the attribute.
        clear_owned_windows_readonly_attribute(pin.path()).unwrap();
        assert!(OpenOptions::new()
            .write(true)
            .share_mode(7)
            .open(pin.path())
            .is_err());
        assert!(fs::remove_file(pin.path()).is_err());
        release_copy(pin, &root);
        release_root(root, source);
    }
}
