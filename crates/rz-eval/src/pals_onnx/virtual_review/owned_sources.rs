//! Caller-owned immutable artifact sources for the V3 physical session owner.
//!
//! The callback can hold original readonly descriptors without an experiments
//! or arena dependency. Callback declarations never replace the independently
//! checked byte budget and digest at the native admission boundary.
use super::super::fail;
use crate::asset;
use crate::error::{BackendError, FailureKind as K, FailureStage as S};
use std::fs::File;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

/// A Linux directory descriptor, never an ordinary path declaration. The
/// bootstrap owner creates its exclusive child with a directory capability;
/// this type holds the resulting descriptor through native session teardown.
/// The descriptor is not permission to clean up any surrounding directory.
#[derive(Clone)]
pub struct VirtualReviewProfileDirectory {
    owner: Arc<ProfileDirectoryOwner>,
}
struct ProfileDirectoryOwner {
    #[cfg(target_os = "linux")]
    file: File,
    path: PathBuf,
    #[cfg(target_os = "linux")]
    device: u64,
    #[cfg(target_os = "linux")]
    inode: u64,
}
impl VirtualReviewProfileDirectory {
    /// Take a real directory descriptor obtained by the startup owner. No
    /// path-only constructor exists. Linux procfs must resolve this exact held
    /// descriptor to the same device/inode; other OS routes are unsupported.
    pub fn from_owned_directory(file: File) -> Result<Self, BackendError> {
        #[cfg(target_os = "linux")]
        {
            use std::os::fd::AsRawFd;
            use std::os::unix::fs::MetadataExt;
            let metadata = file.metadata().map_err(profile_start_error)?;
            let descriptor = file.as_raw_fd();
            if descriptor < 0 || !metadata.is_dir() {
                return Err(profile_error(
                    "V3 profile capability is not a directory descriptor",
                ));
            }
            let value = Self {
                owner: Arc::new(ProfileDirectoryOwner {
                    file,
                    path: PathBuf::from(format!("/proc/self/fd/{descriptor}")),
                    device: metadata.dev(),
                    inode: metadata.ino(),
                }),
            };
            value.validate_live()?;
            Ok(value)
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = file;
            Err(profile_error(
                "owned V3 CUDA profile capability is Linux-only",
            ))
        }
    }
    /// Stable procfd spelling while this owner remains alive. It must never be
    /// canonicalized into a mutable ordinary parent path before profile I/O.
    pub fn path(&self) -> &Path {
        &self.owner.path
    }
    pub(in crate::pals_onnx) fn validate_live(&self) -> Result<(), BackendError> {
        #[cfg(target_os = "linux")]
        {
            use std::os::fd::AsRawFd;
            use std::os::unix::fs::MetadataExt;
            let descriptor = self.owner.file.as_raw_fd();
            let expected = PathBuf::from(format!("/proc/self/fd/{descriptor}"));
            let held = self.owner.file.metadata().map_err(profile_start_error)?;
            let alias = std::fs::metadata(&self.owner.path).map_err(profile_start_error)?;
            let link = std::fs::symlink_metadata(&self.owner.path).map_err(profile_start_error)?;
            if descriptor < 0
                || self.owner.path != expected
                || !held.is_dir()
                || !alias.is_dir()
                || !link.file_type().is_symlink()
                || held.dev() != self.owner.device
                || held.ino() != self.owner.inode
                || alias.dev() != held.dev()
                || alias.ino() != held.ino()
            {
                return Err(profile_error(
                    "V3 profile procfd identity differs from held directory",
                ));
            }
            Ok(())
        }
        #[cfg(not(target_os = "linux"))]
        {
            Err(profile_error(
                "owned V3 CUDA profile capability is Linux-only",
            ))
        }
    }
    pub(in crate::pals_onnx) fn read_finished_profile(
        &self,
        path: &Path,
        role: &str,
        max_bytes: usize,
    ) -> Result<Vec<u8>, BackendError> {
        #[cfg(target_os = "linux")]
        {
            use std::io::Read;
            use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
            self.validate_live()?;
            let exact = path.file_name().map(|name| self.path().join(name));
            if !path.is_absolute()
                || path
                    .components()
                    .any(|part| matches!(part, Component::CurDir | Component::ParentDir))
                || path.parent() != Some(self.path())
                || exact
                    .as_ref()
                    .is_none_or(|expected| path.as_os_str() != expected.as_os_str())
                || !path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| valid_profile_basename(name, role))
            {
                return Err(profile_error(
                    "V3 finished profile is outside its held descriptor role",
                ));
            }
            // Linux O_NOFOLLOW | O_NONBLOCK. The parent is the held directory
            // FD, not a checked-then-reopened ordinary pathname; the final leaf
            // cannot become a symlink or make a FIFO read block indefinitely.
            let mut file = std::fs::OpenOptions::new()
                .read(true)
                .custom_flags(0x20000 | 0x800)
                .open(path)
                .map_err(profile_finish_error)?;
            let before = file.metadata().map_err(profile_finish_error)?;
            if !before.is_file() || before.len() == 0 || before.len() > max_bytes as u64 {
                return Err(profile_error(
                    "V3 finished profile exceeds regular-file budget",
                ));
            }
            let mut bytes = Vec::new();
            bytes
                .try_reserve_exact(before.len() as usize)
                .map_err(|_| profile_error("V3 finished profile allocation failed"))?;
            let limit = (max_bytes as u64)
                .checked_add(1)
                .ok_or_else(|| profile_error("V3 finished profile byte budget overflow"))?;
            (&mut file)
                .take(limit)
                .read_to_end(&mut bytes)
                .map_err(profile_finish_error)?;
            let after = file.metadata().map_err(profile_finish_error)?;
            if bytes.is_empty()
                || bytes.len() > max_bytes
                || before.len() != bytes.len() as u64
                || after.len() != before.len()
                || after.dev() != before.dev()
                || after.ino() != before.ino()
                || after.mtime() != before.mtime()
                || after.mtime_nsec() != before.mtime_nsec()
                || after.ctime() != before.ctime()
                || after.ctime_nsec() != before.ctime_nsec()
            {
                return Err(profile_error(
                    "V3 finished profile changed during held-file read",
                ));
            }
            self.validate_live()?;
            Ok(bytes)
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (path, role, max_bytes);
            Err(profile_error(
                "owned V3 CUDA profile capability is Linux-only",
            ))
        }
    }
}
#[cfg(any(target_os = "linux", test))]
fn valid_profile_basename(name: &str, role: &str) -> bool {
    matches!(role, "public" | "shared_pc" | "shared_warm" | "validator")
        && name.starts_with(&format!("pals-{role}-kernels_"))
        && name.ends_with(".json")
        && !name.contains('/')
        && !name.contains('\\')
}
fn profile_error(message: &'static str) -> BackendError {
    fail(K::BackendUnavailable, S::Backend, message)
}
#[cfg(target_os = "linux")]
fn profile_start_error(error: std::io::Error) -> BackendError {
    profile_error("V3 held profile directory is unavailable")
        .with_external_cause(crate::error::CauseCode::ProfilingStart, &error)
}
#[cfg(target_os = "linux")]
fn profile_finish_error(error: std::io::Error) -> BackendError {
    profile_error("V3 held profile file is unavailable")
        .with_external_cause(crate::error::CauseCode::ProfilingFinish, &error)
}

/// Own every source capability for the full load call. An implementation may
/// read held immutable descriptors; it must not return borrowed native buffers.
/// The checked admission helpers still distrust its returned bytes and budget.
pub trait VirtualReviewOwnedSources: Send + Sync + 'static {
    fn read_manifest(&self, max_bytes: usize) -> Result<Vec<u8>, BackendError>;
    fn read_graph(
        &self,
        role: &str,
        registered_file: &str,
        expected_sha256: [u8; 32],
        max_bytes: usize,
    ) -> Result<Vec<u8>, BackendError>;
}

/// A caller-opened graph file and its registered manifest identity. The file
/// handle is moved into the source owner; no path is reopened during loading.
pub struct VirtualReviewHeldGraph {
    pub role: String,
    pub registered_file: String,
    pub sha256: [u8; 32],
    pub file: File,
}

/// Original manifest and graph descriptors held by a startup custody owner.
/// Construction only transfers the descriptors: it does not authorize a
/// model or establish a physical completion fence. Every read still goes
/// through the ordinary manifest/graph byte and SHA checks at admission.
pub struct VirtualReviewHeldFileSources {
    manifest: File,
    graphs: Vec<VirtualReviewHeldGraph>,
}
impl VirtualReviewHeldFileSources {
    pub fn new(manifest: File, graphs: Vec<VirtualReviewHeldGraph>) -> Self {
        Self { manifest, graphs }
    }
}

#[cfg(unix)]
fn held_read_at(file: &File, bytes: &mut [u8], offset: u64) -> std::io::Result<usize> {
    use std::os::unix::fs::FileExt;
    file.read_at(bytes, offset)
}
/// Positional reads leave the original descriptor's file position unchanged.
/// The subsequent checked boundary verifies the digest of these exact bytes.
#[cfg(unix)]
fn read_held(file: &File, max_bytes: usize) -> Result<Vec<u8>, BackendError> {
    let before = file
        .metadata()
        .map_err(|_| fail(K::Io, S::Asset, "cannot inspect held V3 source"))?;
    let len = usize::try_from(before.len()).map_err(|_| {
        fail(
            K::ResourceExhausted,
            S::Asset,
            "held V3 source exceeds address space",
        )
    })?;
    if !before.is_file() || len == 0 || len > max_bytes {
        return Err(fail(
            K::ResourceExhausted,
            S::Asset,
            "held V3 source is not a bounded regular file",
        ));
    }
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(len).map_err(|_| {
        fail(
            K::ResourceExhausted,
            S::Asset,
            "held V3 source allocation failed",
        )
    })?;
    bytes.resize(len, 0);
    let mut offset = 0;
    while offset < len {
        match held_read_at(file, &mut bytes[offset..], offset as u64) {
            Ok(0) => return Err(fail(K::Io, S::Asset, "held V3 source ended early")),
            Ok(n) => offset += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => return Err(fail(K::Io, S::Asset, "held V3 source read failed")),
        }
    }
    let mut extra = [0_u8; 1];
    if held_read_at(file, &mut extra, len as u64)
        .map_err(|_| fail(K::Io, S::Asset, "cannot check held V3 source length"))?
        != 0
    {
        return Err(fail(K::IdentityMismatch, S::Asset, "held V3 source grew"));
    }
    let after = file
        .metadata()
        .map_err(|_| fail(K::Io, S::Asset, "cannot recheck held V3 source"))?;
    if before.len() != after.len()
        || before
            .modified()
            .ok()
            .zip(after.modified().ok())
            .is_some_and(|(a, b)| a != b)
    {
        return Err(fail(
            K::IdentityMismatch,
            S::Asset,
            "held V3 source changed",
        ));
    }
    Ok(bytes)
}

#[cfg(not(unix))]
fn read_held(_file: &File, _max_bytes: usize) -> Result<Vec<u8>, BackendError> {
    Err(fail(
        K::UnsupportedModel,
        S::Asset,
        "held V3 source positional reads require Unix",
    ))
}

impl VirtualReviewOwnedSources for VirtualReviewHeldFileSources {
    fn read_manifest(&self, max_bytes: usize) -> Result<Vec<u8>, BackendError> {
        read_held(&self.manifest, max_bytes)
    }

    fn read_graph(
        &self,
        role: &str,
        registered_file: &str,
        expected_sha256: [u8; 32],
        max_bytes: usize,
    ) -> Result<Vec<u8>, BackendError> {
        if !matches!(
            Path::new(registered_file)
                .components()
                .collect::<Vec<_>>()
                .as_slice(),
            [Component::Normal(_)]
        ) {
            return Err(fail(
                K::InvalidInput,
                S::Asset,
                "V3 graph must be a basename",
            ));
        }
        let mut matches = self
            .graphs
            .iter()
            .filter(|graph| graph.role == role && graph.registered_file == registered_file);
        let graph = matches.next().ok_or_else(|| {
            fail(
                K::IdentityMismatch,
                S::Asset,
                "held V3 graph source is missing",
            )
        })?;
        if matches.next().is_some() || graph.sha256 != expected_sha256 {
            return Err(fail(
                K::IdentityMismatch,
                S::Asset,
                "held V3 graph identity differs",
            ));
        }
        read_held(&graph.file, max_bytes)
    }
}

/// Compatibility source for the original manifest-relative path loader.
/// Construction only records the path; config errors retain their precedence.
pub(super) struct PathSources {
    manifest_path: PathBuf,
}
impl PathSources {
    pub(super) fn new(path: &Path) -> Self {
        Self {
            manifest_path: path.to_owned(),
        }
    }
    fn validate_path(&self) -> Result<(), BackendError> {
        if !self.manifest_path.is_absolute() {
            return Err(fail(
                K::InvalidInput,
                S::Asset,
                "V3 manifest path must be absolute",
            ));
        }
        Ok(())
    }
}
impl VirtualReviewOwnedSources for PathSources {
    fn read_manifest(&self, max_bytes: usize) -> Result<Vec<u8>, BackendError> {
        self.validate_path()?;
        asset::read_bounded(&self.manifest_path, max_bytes)
    }
    fn read_graph(
        &self,
        _role: &str,
        registered_file: &str,
        _expected_sha256: [u8; 32],
        max_bytes: usize,
    ) -> Result<Vec<u8>, BackendError> {
        self.validate_path()?;
        // Registered graphs are basenames from strict manifest admission.
        // Keep even a direct callback invocation under the original parent.
        if !matches!(
            Path::new(registered_file)
                .components()
                .collect::<Vec<_>>()
                .as_slice(),
            [Component::Normal(_)]
        ) {
            return Err(fail(
                K::InvalidInput,
                S::Asset,
                "V3 registered graph must be a basename",
            ));
        }
        let parent = self
            .manifest_path
            .parent()
            .ok_or_else(|| fail(K::InvalidInput, S::Asset, "V3 export has no parent"))?;
        // The expected digest is conveyed to arbitrary owned implementations.
        // Here the checked boundary hashes once after this bounded path read.
        asset::read_bounded(&parent.join(registered_file), max_bytes)
    }
}

pub(super) fn read_manifest_checked(
    sources: &dyn VirtualReviewOwnedSources,
    expected_sha256: [u8; 32],
    max_bytes: usize,
) -> Result<Vec<u8>, BackendError> {
    let bytes = sources.read_manifest(max_bytes)?;
    if bytes.is_empty() || bytes.len() > max_bytes {
        return Err(fail(
            K::ResourceExhausted,
            S::Asset,
            "V3 manifest source returned empty or over-limit bytes",
        ));
    }
    if asset::sha256(&bytes) != expected_sha256 {
        return Err(fail(
            K::IdentityMismatch,
            S::Asset,
            "V3 manifest digest differs",
        ));
    }
    Ok(bytes)
}

pub(super) fn read_graph_checked(
    sources: &dyn VirtualReviewOwnedSources,
    role: &str,
    registered_file: &str,
    expected_sha256: [u8; 32],
    max_bytes: usize,
) -> Result<(Vec<u8>, [u8; 32]), BackendError> {
    let bytes = sources.read_graph(role, registered_file, expected_sha256, max_bytes)?;
    if bytes.is_empty() || bytes.len() > max_bytes {
        return Err(fail(
            K::ResourceExhausted,
            S::Asset,
            "V3 graph source returned empty or over-limit bytes",
        ));
    }
    let digest = asset::sha256(&bytes);
    if digest != expected_sha256 {
        return Err(fail(
            K::IdentityMismatch,
            S::Asset,
            "V3 graph digest differs",
        ));
    }
    Ok((bytes, digest))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    #[test]
    fn held_sources_read_original_descriptors_without_moving_their_offsets() {
        use std::io::{Seek, SeekFrom, Write};
        use std::sync::atomic::{AtomicU64, Ordering};

        static SEQUENCE: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "rz-held-v3-source-{}-{}",
            std::process::id(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed),
        ));
        struct RemoveFile(PathBuf);
        impl Drop for RemoveFile {
            fn drop(&mut self) {
                let _ = std::fs::remove_file(&self.0);
            }
        }
        let _cleanup = RemoveFile(path.clone());
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        file.write_all(b"original-graph").unwrap();
        file.seek(SeekFrom::Start(3)).unwrap();
        // try_clone shares the original offset on Unix. Positional reads of
        // the held owner must preserve it for the caller and other users.
        let mut observer = file.try_clone().unwrap();
        let graph = file.try_clone().unwrap();
        let digest = asset::sha256(b"original-graph");
        let owner = VirtualReviewHeldFileSources::new(
            file,
            vec![VirtualReviewHeldGraph {
                role: "public".into(),
                registered_file: "public.onnx".into(),
                sha256: digest,
                file: graph,
            }],
        );
        assert_eq!(owner.read_manifest(32).unwrap(), b"original-graph");
        assert_eq!(
            owner
                .read_graph("public", "public.onnx", digest, 32)
                .unwrap(),
            b"original-graph"
        );
        assert_eq!(observer.stream_position().unwrap(), 3);
        assert!(owner.read_manifest(2).is_err());
        assert!(owner
            .read_graph("public", "public.onnx", [0; 32], 32)
            .is_err());
        assert!(owner
            .read_graph("public", "../public.onnx", digest, 32)
            .is_err());
        assert_eq!(observer.stream_position().unwrap(), 3);
    }
    #[test]
    fn descriptor_profile_names_have_exact_role_and_leaf_scope() {
        for role in ["public", "shared_pc", "shared_warm", "validator"] {
            assert!(valid_profile_basename(
                &format!("pals-{role}-kernels_2026-10-13_00-00-00.json"),
                role
            ));
            assert!(!valid_profile_basename(
                "pals-public-kernels_x.json",
                "unregistered"
            ));
            assert!(!valid_profile_basename(
                &format!("pals-{role}-kernels_x.json/other"),
                role
            ));
            assert!(!valid_profile_basename(
                &format!("pals-{role}-kernels_x\\other.json"),
                role
            ));
            assert!(!valid_profile_basename(
                &format!("pals-{role}-kernels_x.txt"),
                role
            ));
        }
        assert!(!valid_profile_basename(
            "pals-shared_pc-kernels_x.json",
            "public"
        ));
        assert!(!valid_profile_basename(
            "pals-public-kernels_x.json",
            "shared_pc"
        ));
    }
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};

    #[derive(Debug, PartialEq)]
    enum ReadCall {
        Manifest {
            max_bytes: usize,
        },
        Graph {
            role: String,
            file: String,
            expected_sha256: [u8; 32],
            max_bytes: usize,
        },
    }
    struct FakeSources {
        bytes: Vec<u8>,
        calls: Mutex<Vec<ReadCall>>,
        owner_dropped: Arc<AtomicBool>,
    }
    impl FakeSources {
        fn new(bytes: &[u8]) -> Self {
            Self {
                bytes: bytes.to_vec(),
                calls: Mutex::new(Vec::new()),
                owner_dropped: Arc::new(AtomicBool::new(false)),
            }
        }
    }
    impl VirtualReviewOwnedSources for FakeSources {
        fn read_manifest(&self, max_bytes: usize) -> Result<Vec<u8>, BackendError> {
            self.calls
                .lock()
                .unwrap()
                .push(ReadCall::Manifest { max_bytes });
            // Deliberately ignores the bound; the native boundary must check.
            Ok(self.bytes.clone())
        }
        fn read_graph(
            &self,
            role: &str,
            registered_file: &str,
            expected_sha256: [u8; 32],
            max_bytes: usize,
        ) -> Result<Vec<u8>, BackendError> {
            self.calls.lock().unwrap().push(ReadCall::Graph {
                role: role.to_owned(),
                file: registered_file.to_owned(),
                expected_sha256,
                max_bytes,
            });
            Ok(self.bytes.clone())
        }
    }
    impl Drop for FakeSources {
        fn drop(&mut self) {
            self.owner_dropped.store(true, Ordering::Release);
        }
    }

    #[test]
    fn actual_returned_bytes_and_all_graph_arguments_are_independently_checked() {
        let sources = FakeSources::new(b"owned fixture bytes");
        let expected = asset::sha256(&sources.bytes);
        let manifest = read_manifest_checked(&sources, expected, 64).unwrap();
        let (graph, actual) =
            read_graph_checked(&sources, "shared_warm", "fixed.onnx", expected, 32).unwrap();
        assert_eq!(manifest, sources.bytes);
        assert_eq!(graph, sources.bytes);
        assert_eq!(actual, expected);
        assert_eq!(
            *sources.calls.lock().unwrap(),
            vec![
                ReadCall::Manifest { max_bytes: 64 },
                ReadCall::Graph {
                    role: "shared_warm".to_owned(),
                    file: "fixed.onnx".to_owned(),
                    expected_sha256: expected,
                    max_bytes: 32,
                }
            ]
        );
    }
    #[test]
    fn callback_ignoring_limit_cannot_admit_empty_oversize_or_zero_budget() {
        for (bytes, limit) in [
            (b"".as_slice(), 8),
            (b"abc".as_slice(), 2),
            (b"x".as_slice(), 0),
        ] {
            let sources = FakeSources::new(bytes);
            let expected = asset::sha256(bytes);
            let manifest = read_manifest_checked(&sources, expected, limit).unwrap_err();
            let graph =
                read_graph_checked(&sources, "public", "public.onnx", expected, limit).unwrap_err();
            for error in [manifest, graph] {
                assert_eq!(error.kind, K::ResourceExhausted);
                assert_eq!(error.stage, S::Asset);
            }
        }
    }
    #[test]
    fn wrong_expected_digest_is_not_authorized_by_a_successful_callback() {
        let sources = FakeSources::new(b"real bytes");
        let wrong = asset::sha256(b"different bytes");
        let manifest = read_manifest_checked(&sources, wrong, 64).unwrap_err();
        let graph = read_graph_checked(&sources, "validator", "role_validator.onnx", wrong, 64)
            .unwrap_err();
        for error in [manifest, graph] {
            assert_eq!(error.kind, K::IdentityMismatch);
            assert_eq!(error.stage, S::Asset);
        }
        assert!(matches!(
            sources.calls.lock().unwrap().last(),
            Some(ReadCall::Graph { expected_sha256, .. }) if *expected_sha256 == wrong
        ));
    }
    #[test]
    fn read_boundary_borrows_the_source_owner_without_releasing_its_capability() {
        let sources: Arc<dyn VirtualReviewOwnedSources> = Arc::new(FakeSources::new(b"x"));
        let expected = asset::sha256(b"x");
        let first = Arc::clone(&sources);
        assert_eq!(
            read_manifest_checked(first.as_ref(), expected, 1).unwrap(),
            b"x"
        );
        assert_eq!(Arc::strong_count(&sources), 2);
        drop(first);
        assert_eq!(Arc::strong_count(&sources), 1);
        let (graph, _) =
            read_graph_checked(sources.as_ref(), "public", "public.onnx", expected, 1).unwrap();
        assert_eq!(graph, b"x");
        let concrete = FakeSources::new(b"y");
        let owner_dropped = Arc::clone(&concrete.owner_dropped);
        read_manifest_checked(&concrete, asset::sha256(b"y"), 1).unwrap();
        assert!(!owner_dropped.load(Ordering::Acquire));
        drop(concrete);
        assert!(owner_dropped.load(Ordering::Acquire));
    }
    #[test]
    fn relative_path_validation_is_deferred_and_does_not_open_a_file() {
        let sources = PathSources::new(Path::new("relative-export.json"));
        assert_eq!(sources.manifest_path, PathBuf::from("relative-export.json"));
        let manifest = sources.read_manifest(128).unwrap_err();
        let graph = sources
            .read_graph("public", "public_memory.onnx", [1; 32], 128)
            .unwrap_err();
        for error in [manifest, graph] {
            assert_eq!(error.kind, K::InvalidInput);
            assert_eq!(error.stage, S::Asset);
            assert_eq!(error.detail, "V3 manifest path must be absolute");
        }
    }
}
