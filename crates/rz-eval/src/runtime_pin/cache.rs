//! Immutable runtime storage, separate from per-run evidence and evaluation caches.
//! The caller owns the root and ancestors. This is not a same-UID sandbox.

use super::*;
use std::{
    io::Read,
    path::Component,
    thread,
    time::{Duration, Instant},
};

const MAX_ENTRIES: usize = 4;
const MAX_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const LOCK_WAIT: Duration = Duration::from_secs(60);
const LOCK_NAME: &str = ".publish-lock";

/// A bounded, caller-owned cache outside Git. Published entries are never
/// replaced, repaired or automatically evicted: another process may map them.
#[derive(Clone)]
pub struct RuntimeCache {
    root: PathBuf,
    max_entries: usize,
    max_bytes: u64,
}

impl std::fmt::Debug for RuntimeCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuntimeCache")
            .field("max_entries", &self.max_entries)
            .field("max_bytes", &self.max_bytes)
            .finish_non_exhaustive()
    }
}

impl RuntimeCache {
    /// Explicit local handoff to a child bootstrap. Do not render this private
    /// path in shared diagnostics or artifact identities. Opening the cache
    /// does not issue any library pin or authorize native code execution.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// No run ID or PID in the default slot. Platform/architecture and the
    /// entry content key isolate incompatible binaries. CI uses RUNNER_TEMP.
    pub fn for_user() -> Result<Self, BackendError> {
        let base = if let Some(base) = std::env::var_os("RUNNER_TEMP") {
            PathBuf::from(base).join("RoveZero").join("cache")
        } else if cfg!(windows) {
            PathBuf::from(
                std::env::var_os("APPDATA")
                    .ok_or_else(|| io_error("runtime cache requires APPDATA"))?,
            )
            .join("RoveZero")
            .join("cache")
        } else {
            let base = match std::env::var_os("XDG_CACHE_HOME") {
                Some(base) => PathBuf::from(base),
                None => PathBuf::from(
                    std::env::var_os("HOME")
                        .ok_or_else(|| io_error("runtime cache requires HOME or XDG_CACHE_HOME"))?,
                )
                .join(".cache"),
            };
            base.join("rovezero")
        };
        Self::open(&base.join("native-runtime-v1").join(format!(
            "{}-{}",
            std::env::consts::OS,
            std::env::consts::ARCH
        )))
    }

    /// Creates missing directories privately; rejects links, relative paths,
    /// parent traversal, Git roots and existing group/world-writable leaf roots.
    /// This path is a cache slot, never a run/report/source directory.
    pub fn open(root: &Path) -> Result<Self, BackendError> {
        if !root.is_absolute()
            || root
                .components()
                .any(|part| matches!(part, Component::ParentDir | Component::CurDir))
        {
            return Err(io_error(
                "runtime cache root must be an absolute owned path",
            ));
        }
        reject_git_ancestors(root)?;
        let mut current = PathBuf::new();
        for part in root.components() {
            current.push(part.as_os_str());
            // A Windows drive/UNC/verbatim prefix alone is not an absolute
            // directory. Inspect it only after appending its root separator.
            if matches!(part, Component::Prefix(_)) {
                continue;
            }
            match fs::symlink_metadata(&current) {
                Ok(metadata) if metadata.is_dir() && !is_link(&metadata) => {}
                Ok(_) => {
                    return Err(io_error(
                        "runtime cache path contains a link or nondirectory",
                    ))
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    match private_directory(&current) {
                        Ok(()) => {}
                        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                            check_directory(&current, false)?;
                        }
                        Err(error) => {
                            return Err(cache_io("cannot create runtime cache directory", &error))
                        }
                    }
                }
                Err(error) => return Err(cache_io("runtime cache path metadata failed", &error)),
            }
        }
        check_directory(root, false)?;
        let root = root
            .canonicalize()
            .map_err(|error| cache_io("cannot resolve runtime cache root", &error))?;
        reject_git_ancestors(&root)?;
        Ok(Self {
            root,
            max_entries: MAX_ENTRIES,
            max_bytes: MAX_BYTES,
        })
    }

    /// A hit hashes and pins the cached bytes, without copying or reading the
    /// mutable source. The explicitly supplied digest remains authoritative.
    pub fn library(
        &self,
        source: &Path,
        expected_sha256: &str,
    ) -> Result<RuntimeLibraryPin, BackendError> {
        check_directory(&self.root, false)?;
        let expected = asset::parse_sha256(expected_sha256)?;
        let filename = source
            .file_name()
            .and_then(|name| name.to_str())
            .filter(|name| {
                !name.is_empty() && name.len() <= 255 && !name.contains(['\\', '/', '\n', '\r'])
            })
            .ok_or_else(|| io_error("runtime cache source requires a bounded library basename"))?;
        let mut digest = Sha256::new();
        digest.update(b"RoveZero/native-cpu-cache\0v1\0");
        digest.update((filename.len() as u32).to_be_bytes());
        digest.update(filename.as_bytes());
        digest.update(expected);
        let key: [u8; 32] = digest.finalize().into();
        let entry = self.root.join(format!("cpu-{}", hex(key)));
        let read = |origin| read_cpu(&entry, filename, expected, key, origin);
        if exists(&entry)? {
            return read(RuntimeStorageOrigin::CacheReused);
        }
        let _lock = match self.lock(LOCK_WAIT) {
            Ok(lock) => lock,
            Err(error) => {
                return reuse_after_lock_error(&entry, error, || {
                    read(RuntimeStorageOrigin::CacheReused)
                })
            }
        };
        if exists(&entry)? {
            return read(RuntimeStorageOrigin::CacheReused);
        }
        let metadata = fs::symlink_metadata(source)
            .map_err(|error| cache_io("runtime cache source metadata failed", &error))?;
        if !metadata.is_file()
            || is_link(&metadata)
            || metadata.len() == 0
            || metadata.len() > MAX_LIBRARY_BYTES as u64
        {
            return Err(io_error(
                "runtime cache source must be a bounded regular file",
            ));
        }
        let bytes = metadata.len();
        self.check_budget(bytes)?;
        let pin = RuntimeLibraryPin::copy_verified(source, &self.root, expected_sha256)?;
        let stage = Unpublished::from_pin(&pin)?;
        let actual_bytes = pin.storage().bytes;
        drop(pin); // Windows denies rename while a file pin is held.
        if actual_bytes != bytes {
            return Err(identity("runtime cache source size changed during copy"));
        }
        stage.publish(&entry)?;
        read(RuntimeStorageOrigin::CacheCreated)
    }

    /// Reject historical CUDA bundle cache execution before reading the cache
    /// or source. Bundle metadata can still be parsed independently.
    pub fn cuda_bundle(
        &self,
        source_root: &Path,
        spec: &CudaRuntimeBundleSpec,
    ) -> Result<RuntimeLibraryPin, BackendError> {
        self.cuda_bundle_observed(source_root, spec, &mut |_| Ok(()))
    }

    pub(super) fn cuda_bundle_observed(
        &self,
        source_root: &Path,
        spec: &CudaRuntimeBundleSpec,
        checkpoint: &mut dyn FnMut(u64) -> Result<(), BackendError>,
    ) -> Result<RuntimeLibraryPin, BackendError> {
        refuse_legacy_cuda_bundle()?;
        checkpoint(0)?;
        check_directory(&self.root, false)?;
        let key = spec.digest()?;
        #[cfg(target_os = "linux")]
        {
            let prefix = if spec.schema_version == 2 {
                "cuda-alias-v2"
            } else {
                "cuda"
            };
            let entry = self.root.join(format!("{prefix}-{}", hex(key)));
            if exists(&entry)? {
                return read_cuda(
                    &entry,
                    spec,
                    key,
                    RuntimeStorageOrigin::CacheReused,
                    checkpoint,
                );
            }
            let _lock = match self.lock_observed(LOCK_WAIT, checkpoint) {
                Ok(lock) => lock,
                // A caller deadline/cancel/read rejection is terminal. Only an
                // ordinary publication race may reuse a completed entry.
                Err((error, true)) => return Err(error),
                Err((error, false)) => {
                    return reuse_after_lock_error(&entry, error, || {
                        checkpoint(0)?;
                        read_cuda(
                            &entry,
                            spec,
                            key,
                            RuntimeStorageOrigin::CacheReused,
                            checkpoint,
                        )
                    })
                }
            };
            checkpoint(0)?;
            if exists(&entry)? {
                return read_cuda(
                    &entry,
                    spec,
                    key,
                    RuntimeStorageOrigin::CacheReused,
                    checkpoint,
                );
            }
            let incoming = spec.files.iter().map(|file| file.bytes).sum::<u64>()
                + spec
                    .loader_aliases
                    .as_deref()
                    .unwrap_or(&[])
                    .iter()
                    .map(|alias| alias.target.len() as u64)
                    .sum::<u64>();
            self.check_budget(incoming)?;
            checkpoint(0)?;
            let pin = RuntimeLibraryPin::copy_linux_cuda_bundle_observed(
                source_root,
                &self.root,
                spec,
                checkpoint,
            )?;
            let stage = Unpublished::from_pin(&pin)?;
            drop(pin);
            checkpoint(0)?;
            stage.publish(&entry)?;
            checkpoint(0)?;
            read_cuda(
                &entry,
                spec,
                key,
                RuntimeStorageOrigin::CacheCreated,
                checkpoint,
            )
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (source_root, key);
            Err(BackendError::new(
                K::BackendUnavailable,
                S::Backend,
                "CUDA runtime bundle pins currently require Linux",
            ))
        }
    }

    fn lock(&self, timeout: Duration) -> Result<PublishLock, BackendError> {
        self.lock_observed(timeout, &mut |_| Ok(()))
            .map_err(|(error, _)| error)
    }

    fn lock_observed(
        &self,
        timeout: Duration,
        checkpoint: &mut dyn FnMut(u64) -> Result<(), BackendError>,
    ) -> Result<PublishLock, (BackendError, bool)> {
        check_directory(&self.root, false).map_err(|error| (error, false))?;
        let path = self.root.join(LOCK_NAME);
        let until = Instant::now() + timeout;
        loop {
            checkpoint(0).map_err(|error| (error, true))?;
            match private_directory(&path) {
                Ok(()) => return Ok(PublishLock(path)),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    check_directory(&path, false).map_err(|error| (error, false))?;
                    if Instant::now() >= until {
                        return Err((BackendError::new(K::ResourceExhausted, S::Backend,
                            "runtime cache publication is busy; inspect an inactive crashed owner before removing its lock"), false));
                    }
                    thread::sleep(
                        Duration::from_millis(25)
                            .min(until.saturating_duration_since(Instant::now())),
                    );
                }
                Err(error) => {
                    return Err((
                        cache_io("cannot lock runtime cache publication", &error),
                        false,
                    ))
                }
            }
        }
    }

    fn check_budget(&self, incoming: u64) -> Result<(), BackendError> {
        let mut entries = 0;
        let mut bytes = incoming;
        for entry in fs::read_dir(&self.root)
            .map_err(|error| cache_io("cannot inspect runtime cache budget", &error))?
        {
            let entry =
                entry.map_err(|error| cache_io("runtime cache entry metadata failed", &error))?;
            if entry.file_name() == LOCK_NAME {
                continue;
            }
            let name = entry.file_name();
            let name = name
                .to_str()
                .ok_or_else(|| io_error("runtime cache contains an unknown entry"))?;
            let key = name
                .strip_prefix("cpu-")
                .or_else(|| name.strip_prefix("cuda-alias-v2-"))
                .or_else(|| name.strip_prefix("cuda-"));
            if !key.is_some_and(|key| {
                key.len() == 64
                    && key
                        .bytes()
                        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            }) {
                return Err(io_error("runtime cache contains an unpublished or unknown entry; preserve it for owned cleanup"));
            }
            entries += 1;
            if entries >= self.max_entries {
                return Err(capacity());
            }
            check_directory(&entry.path(), true)?;
            let mut files = 0;
            for file in fs::read_dir(entry.path())
                .map_err(|error| cache_io("cannot inspect runtime cache files", &error))?
            {
                files += 1;
                if files > MAX_BUNDLE_FILES {
                    return Err(capacity());
                }
                let file =
                    file.map_err(|error| cache_io("runtime cache file metadata failed", &error))?;
                let metadata = fs::symlink_metadata(file.path())
                    .map_err(|error| cache_io("runtime cache file metadata failed", &error))?;
                if is_link(&metadata) {
                    #[cfg(target_os = "linux")]
                    {
                        if !name.starts_with("cuda-alias-v2-") {
                            return Err(identity("runtime cache contains an unexpected file type"));
                        }
                        let basename = file.file_name();
                        let basename = basename
                            .to_str()
                            .ok_or_else(|| identity("runtime cache alias basename is invalid"))?;
                        aliases::verify_cached_name(&entry.path(), basename)?;
                    }
                    #[cfg(not(target_os = "linux"))]
                    return Err(identity("runtime cache contains an unexpected file type"));
                } else if !metadata.is_file() {
                    return Err(identity("runtime cache contains an unexpected file type"));
                }
                if name.starts_with("cuda-alias-v2-") && metadata.is_file() {
                    let basename = file.file_name();
                    if !basename
                        .to_str()
                        .is_some_and(aliases::allowed_original_basename)
                    {
                        return Err(identity(
                            "runtime alias cache contains an undeclared regular file",
                        ));
                    }
                }
                #[cfg(unix)]
                {
                    use std::os::unix::fs::MetadataExt;
                    if metadata.nlink() != 1 {
                        return Err(identity("runtime cache contains a hard-linked file"));
                    }
                }
                bytes = bytes.checked_add(metadata.len()).ok_or_else(capacity)?;
                if bytes > self.max_bytes {
                    return Err(capacity());
                }
            }
        }
        if bytes > self.max_bytes {
            return Err(capacity());
        }
        Ok(())
    }
}

fn capacity() -> BackendError {
    BackendError::new(
        K::ResourceExhausted,
        S::Backend,
        "runtime cache capacity reached; remove only verified inactive entries",
    )
}
fn identity(detail: &'static str) -> BackendError {
    BackendError::new(K::IdentityMismatch, S::Backend, detail)
}
fn cache_io(detail: &'static str, error: &std::io::Error) -> BackendError {
    io_error(detail).with_external_cause(crate::error::CauseCode::RuntimePath, error)
}
// A creator may publish and release its directory lock between a contender's
// create/check calls. Windows can also report a directory pending deletion as
// an I/O error. Only the existing full immutable-entry verifier may recover;
// no new publication, lock stealing, cache repair or indefinite retry occurs.
fn reuse_after_lock_error(
    entry: &Path,
    original: BackendError,
    read: impl FnOnce() -> Result<RuntimeLibraryPin, BackendError>,
) -> Result<RuntimeLibraryPin, BackendError> {
    if !matches!(exists(entry), Ok(true)) {
        return Err(original);
    }
    // BackendError formats bounded static detail and coded/hashed causes,
    // never the private path or raw external error. Preserve the original
    // failure even if the completed publication validates successfully.
    eprintln!("RoveZero runtime cache: {original}; rechecking a published entry");
    read()
}
fn hex(key: [u8; 32]) -> String {
    key.iter().map(|byte| format!("{byte:02x}")).collect()
}
fn is_link(metadata: &fs::Metadata) -> bool {
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        metadata.file_attributes() & 0x400 != 0
    }
    #[cfg(not(windows))]
    {
        metadata.file_type().is_symlink()
    }
}
fn reject_git_ancestors(path: &Path) -> Result<(), BackendError> {
    for ancestor in path.ancestors() {
        match fs::symlink_metadata(ancestor.join(".git")) {
            Ok(_) => return Err(io_error("runtime cache must be outside Git checkouts")),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(cache_io("cannot verify runtime cache Git boundary", &error)),
        }
    }
    Ok(())
}
fn private_directory(path: &Path) -> std::io::Result<()> {
    let builder = fs::DirBuilder::new();
    #[cfg(unix)]
    let builder = {
        use std::os::unix::fs::DirBuilderExt;
        let mut builder = builder;
        builder.mode(0o700);
        builder
    };
    builder.create(path)
}
fn check_directory(path: &Path, sealed: bool) -> Result<(), BackendError> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| cache_io("runtime cache directory is unavailable", &error))?;
    if !metadata.is_dir() || is_link(&metadata) {
        return Err(identity(
            "runtime cache directory is a link or nondirectory",
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = metadata.permissions().mode() & 0o777;
        if (sealed && mode != 0o500) || (!sealed && mode & 0o077 != 0) {
            return Err(identity(
                "runtime cache directory permissions differ from its private ownership contract",
            ));
        }
    }
    #[cfg(not(unix))]
    let _ = sealed;
    Ok(())
}
fn exists(path: &Path) -> Result<bool, BackendError> {
    match fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(cache_io("runtime cache entry lookup failed", &error)),
    }
}
struct PublishLock(PathBuf);
impl Drop for PublishLock {
    fn drop(&mut self) {
        let _ = fs::remove_dir(&self.0);
    }
}

/// Owns exactly the newly created files, never a previously published entry.
struct Unpublished {
    directory: PathBuf,
    filenames: Vec<PathBuf>,
    published: bool,
}
impl Unpublished {
    fn from_pin(pin: &RuntimeLibraryPin) -> Result<Self, BackendError> {
        let directory = pin
            .path()
            .parent()
            .ok_or_else(|| io_error("runtime staging copy lacks its directory"))?
            .to_owned();
        let mut filenames: Vec<_> = match pin.bundle_files() {
            Some(files) => files
                .iter()
                .map(|file| directory.join(&file.filename))
                .collect(),
            None => vec![pin.path().to_owned()],
        };
        if let Some(bundle) = &pin.0.bundle {
            filenames.extend(
                bundle
                    .loader_aliases
                    .as_deref()
                    .unwrap_or(&[])
                    .iter()
                    .map(|alias| directory.join(&alias.filename)),
            );
        }
        Ok(Self {
            directory,
            filenames,
            published: false,
        })
    }
    fn publish(mut self, entry: &Path) -> Result<(), BackendError> {
        if exists(entry)? {
            return Err(identity(
                "runtime cache destination appeared during exclusive publication",
            ));
        }
        fs::rename(&self.directory, entry)
            .map_err(|error| cache_io("cannot publish verified runtime cache entry", &error))?;
        self.published = true;
        Ok(())
    }
}
impl Drop for Unpublished {
    fn drop(&mut self) {
        if self.published {
            return;
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = fs::set_permissions(&self.directory, fs::Permissions::from_mode(0o700));
        }
        for path in &self.filenames {
            #[cfg(windows)]
            let _ = clear_owned_windows_readonly_attribute(path);
            let _ = fs::remove_file(path);
        }
        let _ = fs::remove_dir(&self.directory);
    }
}

fn entry_names(directory: &Path, names: &[&str]) -> Result<(), BackendError> {
    check_directory(directory, true)?;
    let mut count = 0;
    for file in fs::read_dir(directory)
        .map_err(|error| cache_io("cannot verify runtime cache directory contents", &error))?
    {
        let file =
            file.map_err(|error| cache_io("runtime cache directory contents failed", &error))?;
        count += 1;
        if count > names.len()
            || !file
                .file_name()
                .to_str()
                .is_some_and(|name| names.contains(&name))
        {
            return Err(identity("runtime cache contains undeclared files"));
        }
    }
    if count != names.len() {
        return Err(identity("runtime cache is missing declared files"));
    }
    Ok(())
}

fn pinned_file(
    path: &Path,
    expected_bytes: Option<u64>,
    expected_digest: [u8; 32],
) -> Result<File, BackendError> {
    pinned_file_observed(path, expected_bytes, expected_digest, &mut |_| Ok(()))
}

fn pinned_file_observed(
    path: &Path,
    expected_bytes: Option<u64>,
    expected_digest: [u8; 32],
    checkpoint: &mut dyn FnMut(u64) -> Result<(), BackendError>,
) -> Result<File, BackendError> {
    checkpoint(0)?;
    let named = fs::symlink_metadata(path)
        .map_err(|error| cache_io("runtime cached file is unavailable", &error))?;
    if !named.is_file() || is_link(&named) {
        return Err(identity(
            "runtime cached file must be a regular nonsymlink file",
        ));
    }
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(0x20000 | 0x800);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        options.share_mode(1).custom_flags(0x00200000);
    }
    let mut file = options
        .open(path)
        .map_err(|error| cache_io("cannot pin runtime cached file", &error))?;
    let opened = file
        .metadata()
        .map_err(|error| cache_io("runtime cached pin metadata failed", &error))?;
    if !opened.is_file()
        || is_link(&opened)
        || opened.len() == 0
        || opened.len() > expected_bytes.unwrap_or(MAX_LIBRARY_BYTES as u64)
        || expected_bytes.is_some_and(|bytes| bytes != opened.len())
    {
        return Err(identity("runtime cached file size or type differs"));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};
        if opened.dev() != named.dev()
            || opened.ino() != named.ino()
            || opened.nlink() != 1
            || opened.permissions().mode() & 0o777 != 0o400
        {
            return Err(identity(
                "runtime cached inode, link count or readonly permissions differ",
            ));
        }
    }
    #[cfg(windows)]
    if !opened.permissions().readonly() {
        return Err(identity("runtime cached file is not readonly"));
    }
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    let mut remaining = opened.len();
    while remaining > 0 {
        checkpoint(remaining.min(64 * 1024))?;
        let count = file
            .read(&mut buffer[..remaining.min(64 * 1024) as usize])
            .map_err(|error| cache_io("cannot verify runtime cached bytes", &error))?;
        if count == 0 {
            return Err(identity("runtime cached bytes ended before declared size"));
        }
        digest.update(&buffer[..count]);
        remaining -= count as u64;
    }
    checkpoint(1)?;
    if file
        .read(&mut buffer[..1])
        .map_err(|error| cache_io("cannot verify runtime cached end", &error))?
        != 0
        || <[u8; 32]>::from(digest.finalize()) != expected_digest
    {
        return Err(identity("runtime cached digest differs"));
    }
    use std::io::{Seek, SeekFrom};
    file.seek(SeekFrom::Start(0))
        .map_err(|error| cache_io("cannot rewind runtime cached pin", &error))?;
    Ok(file)
}

fn read_cpu(
    entry: &Path,
    filename: &str,
    expected: [u8; 32],
    key: [u8; 32],
    origin: RuntimeStorageOrigin,
) -> Result<RuntimeLibraryPin, BackendError> {
    entry_names(entry, &[filename])?;
    let path = entry.join(filename);
    let pin = pinned_file(&path, None, expected)?;
    let bytes = pin
        .metadata()
        .map_err(|error| cache_io("runtime cached size is unavailable", &error))?
        .len();
    Ok(RuntimeLibraryPin(Arc::new(OwnedCopy {
        path,
        digest: expected,
        _pin: pin,
        bundle: None,
        storage: RuntimeStorage {
            origin,
            cache_key: Some(key),
            bytes,
        },
    })))
}

#[cfg(target_os = "linux")]
fn read_cuda(
    entry: &Path,
    spec: &CudaRuntimeBundleSpec,
    key: [u8; 32],
    origin: RuntimeStorageOrigin,
    checkpoint: &mut dyn FnMut(u64) -> Result<(), BackendError>,
) -> Result<RuntimeLibraryPin, BackendError> {
    checkpoint(0)?;
    entry_names(
        entry,
        &spec
            .files
            .iter()
            .map(|file| file.filename.as_str())
            .chain(
                spec.loader_aliases
                    .as_deref()
                    .unwrap_or(&[])
                    .iter()
                    .map(|alias| alias.filename.as_str()),
            )
            .collect::<Vec<_>>(),
    )?;
    let mut files = spec.files.clone();
    files.sort_unstable_by(|left, right| left.filename.cmp(&right.filename));
    let mut core = None;
    let mut pins = Vec::with_capacity(files.len() - 1);
    for file in &files {
        let path = entry.join(&file.filename);
        let digest = bundle_file_digest(file)?;
        let pin = pinned_file_observed(&path, Some(file.bytes), digest, checkpoint)?;
        if file.role == RuntimeBundleFileRole::Core {
            core = Some((path, digest, pin));
        } else {
            pins.push(pin);
        }
    }
    aliases::verify(entry, spec.loader_aliases.as_deref().unwrap_or(&[]), &files)?;
    let (path, digest, pin) =
        core.ok_or_else(|| bundle_error("cached CUDA core pin is missing"))?;
    Ok(RuntimeLibraryPin(Arc::new(OwnedCopy {
        path,
        digest,
        _pin: pin,
        bundle: Some(OwnedCudaBundle {
            digest: key,
            schema_version: spec.schema_version,
            files,
            loader_aliases: spec.loader_aliases.clone(),
            _pins: pins,
        }),
        storage: RuntimeStorage {
            origin,
            cache_key: Some(key),
            bytes: spec.files.iter().map(|file| file.bytes).sum(),
        },
    })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::{Command, Stdio};
    static NEXT_TEST: AtomicU64 = AtomicU64::new(0);
    const PAYLOAD: &[u8] = b"authored tiny runtime-cache fixture; never executes native code";

    struct Fixture {
        root: PathBuf,
        source: PathBuf,
        cache: RuntimeCache,
    }
    impl Fixture {
        fn new() -> Self {
            #[cfg(windows)]
            let base = PathBuf::from(std::env::var_os("APPDATA").expect("APPDATA required"));
            #[cfg(not(windows))]
            let base = std::env::var_os("RUNNER_TEMP")
                .map(PathBuf::from)
                .unwrap_or_else(std::env::temp_dir);
            let root = base.join("RoveZero/tmp/runtime-cache-tests").join(format!(
                "{}-{}",
                std::process::id(),
                NEXT_TEST.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(&root).unwrap();
            let root = root.canonicalize().unwrap();
            let source = root.join("library.fixture");
            fs::write(&source, PAYLOAD).unwrap();
            let cache = RuntimeCache::open(&root.join("cache")).unwrap();
            Self {
                root,
                source,
                cache,
            }
        }
        fn load(&self) -> RuntimeLibraryPin {
            self.cache
                .library(&self.source, &asset::hex_sha256(PAYLOAD))
                .unwrap()
        }
    }
    fn make_writable(path: &Path, directory: bool) {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(
                path,
                fs::Permissions::from_mode(if directory { 0o700 } else { 0o600 }),
            )
            .unwrap();
        }
        #[cfg(windows)]
        {
            let _ = directory;
            clear_owned_windows_readonly_attribute(path).unwrap();
        }
    }
    // Only the test's fresh root and up to two levels of authored regular files.
    fn release_directory(path: &Path) {
        make_writable(path, true);
        for entry in fs::read_dir(path).unwrap() {
            let entry = entry.unwrap();
            let metadata = fs::symlink_metadata(entry.path()).unwrap();
            if metadata.is_dir() && !is_link(&metadata) {
                release_directory(&entry.path());
            } else {
                if !is_link(&metadata) {
                    make_writable(&entry.path(), false);
                }
                fs::remove_file(entry.path()).unwrap();
            }
        }
        fs::remove_dir(path).unwrap();
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            release_directory(&self.root);
        }
    }

    #[test]
    fn create_then_reuse_pins_the_same_bytes_after_source_mutation() {
        let fixture = Fixture::new();
        let first = fixture.load();
        assert_eq!(first.storage().origin, RuntimeStorageOrigin::CacheCreated);
        fs::write(
            &fixture.source,
            b"mutable source changed after initial publication",
        )
        .unwrap();
        let second = fixture.load();
        assert_eq!(second.storage().origin, RuntimeStorageOrigin::CacheReused);
        assert_eq!(first.path(), second.path());
        assert_eq!(first.binary_digest(), second.binary_digest());
        assert_eq!(first.storage().cache_key, second.storage().cache_key);
        assert_eq!(second.storage().bytes, PAYLOAD.len() as u64);
        assert_eq!(fs::read(second.path()).unwrap(), PAYLOAD);
        assert_eq!(fs::read_dir(&fixture.cache.root).unwrap().count(), 1);
        assert!(!format!("{:?}", fixture.cache).contains(fixture.root.to_str().unwrap()));
        assert!(!format!("{second:?}").contains(fixture.root.to_str().unwrap()));
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            assert_eq!(
                first.0._pin.metadata().unwrap().ino(),
                second.0._pin.metadata().unwrap().ino()
            );
        }
    }

    #[test]
    fn hash_size_missing_extra_and_writable_corruption_never_repairs_a_published_entry() {
        for kind in ["hash", "size", "missing", "extra", "writable"] {
            let fixture = Fixture::new();
            let pin = fixture.load();
            let library = pin.path().to_owned();
            let entry = library.parent().unwrap().to_owned();
            drop(pin);
            make_writable(&entry, true);
            make_writable(&library, false);
            match kind {
                "hash" => {
                    fs::write(&library, vec![b'!'; PAYLOAD.len()]).unwrap();
                }
                "size" => {
                    fs::write(&library, b"short").unwrap();
                }
                "missing" => {
                    fs::remove_file(&library).unwrap();
                }
                "extra" => {
                    fs::write(entry.join("undeclared.fixture"), b"extra").unwrap();
                }
                "writable" => {}
                _ => unreachable!(),
            }
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(&entry, fs::Permissions::from_mode(0o500)).unwrap();
                if kind != "missing" && kind != "writable" {
                    fs::set_permissions(&library, fs::Permissions::from_mode(0o400)).unwrap();
                }
            }
            #[cfg(windows)]
            if kind != "missing" && kind != "writable" {
                let mut p = fs::metadata(&library).unwrap().permissions();
                p.set_readonly(true);
                fs::set_permissions(&library, p).unwrap();
            }
            assert_eq!(
                fixture
                    .cache
                    .library(&fixture.source, &asset::hex_sha256(PAYLOAD))
                    .unwrap_err()
                    .kind,
                K::IdentityMismatch
            );
            assert_eq!(fs::read_dir(&fixture.cache.root).unwrap().count(), 1);
            assert_eq!(fs::read(&fixture.source).unwrap(), PAYLOAD);
        }
    }

    #[test]
    fn bad_source_and_capacity_rejection_leave_no_unpublished_files_or_lock() {
        let fixture = Fixture::new();
        assert_eq!(
            fixture
                .cache
                .library(&fixture.source, &"01".repeat(32))
                .unwrap_err()
                .kind,
            K::IdentityMismatch
        );
        assert_eq!(fs::read_dir(&fixture.cache.root).unwrap().count(), 0);
        for index in 0..MAX_ENTRIES {
            let bytes = format!("authored variant {index}");
            fs::write(&fixture.source, &bytes).unwrap();
            drop(
                fixture
                    .cache
                    .library(&fixture.source, &asset::hex_sha256(bytes.as_bytes()))
                    .unwrap(),
            );
        }
        fs::write(&fixture.source, PAYLOAD).unwrap();
        assert_eq!(
            fixture
                .cache
                .library(&fixture.source, &asset::hex_sha256(PAYLOAD))
                .unwrap_err()
                .kind,
            K::ResourceExhausted
        );
        assert_eq!(
            fs::read_dir(&fixture.cache.root).unwrap().count(),
            MAX_ENTRIES
        );
        let tiny = RuntimeCache {
            max_bytes: 1,
            ..fixture.cache.clone()
        };
        assert_eq!(
            tiny.library(&fixture.source, &asset::hex_sha256(PAYLOAD))
                .unwrap_err()
                .kind,
            K::ResourceExhausted
        );
    }

    #[test]
    fn private_roots_reject_relative_paths_and_git_before_creation() {
        let fixture = Fixture::new();
        assert!(RuntimeCache::open(Path::new("relative-cache")).is_err());
        let checkout = fixture.root.join("checkout");
        fs::create_dir(&checkout).unwrap();
        fs::write(checkout.join(".git"), b"authored fake git boundary").unwrap();
        assert!(RuntimeCache::open(&checkout.join("new-cache")).is_err());
        assert!(!checkout.join("new-cache").exists());
    }

    #[test]
    fn busy_lock_times_out_without_stealing_or_removing_another_creator() {
        let fixture = Fixture::new();
        let lock = fixture.cache.lock(Duration::ZERO).unwrap();
        assert_eq!(
            fixture
                .cache
                .lock(Duration::from_millis(30))
                .err()
                .unwrap()
                .kind,
            K::ResourceExhausted
        );
        assert!(lock.0.is_dir());
        drop(lock);
        drop(fixture.cache.lock(Duration::ZERO).unwrap());
        assert_eq!(fs::read_dir(&fixture.cache.root).unwrap().count(), 0);
    }

    #[test]
    fn lock_failure_reuses_only_a_fully_verified_published_entry() {
        let fixture = Fixture::new();
        let original = BackendError::new(K::Io, S::Backend, "injected publication lock failure");
        let missing = fixture.cache.root.join("not-published");
        let unchanged = reuse_after_lock_error(&missing, original.clone(), || {
            panic!("an absent entry cannot recover the lock failure")
        })
        .unwrap_err();
        assert_eq!(unchanged, original);
        let first = fixture.load();
        let library = first.path().to_owned();
        let entry = library.parent().unwrap().to_owned();
        let reuse =
            reuse_after_lock_error(&entry, original.clone(), || Ok(fixture.load())).unwrap();
        assert_eq!(reuse.storage().origin, RuntimeStorageOrigin::CacheReused);
        assert_eq!(reuse.path(), first.path());
        assert_eq!(reuse.binary_digest(), first.binary_digest());
        drop(reuse);
        drop(first);
        make_writable(&library, false);
        fs::write(&library, b"short corrupt publication").unwrap();
        let error = reuse_after_lock_error(&entry, original, || {
            fixture
                .cache
                .library(&fixture.source, &asset::hex_sha256(PAYLOAD))
        })
        .unwrap_err();
        assert_eq!(error.kind, K::IdentityMismatch);
        assert_eq!(fs::read(&library).unwrap(), b"short corrupt publication");
        assert!(!fixture.cache.root.join(LOCK_NAME).exists());
    }

    #[test]
    fn simultaneous_threads_publish_one_entry_and_keep_independent_pins() {
        let fixture = Fixture::new();
        let barrier = Arc::new(std::sync::Barrier::new(4));
        let workers: Vec<_> = (0..4)
            .map(|_| {
                let cache = fixture.cache.clone();
                let source = fixture.source.clone();
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();
                    cache.library(&source, &asset::hex_sha256(PAYLOAD)).unwrap()
                })
            })
            .collect();
        let pins: Vec<_> = workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect();
        assert_eq!(
            pins.iter()
                .filter(|pin| pin.storage().origin == RuntimeStorageOrigin::CacheCreated)
                .count(),
            1
        );
        assert!(pins.iter().all(|pin| pin.path() == pins[0].path()));
        assert_eq!(fs::read_dir(&fixture.cache.root).unwrap().count(), 1);
    }

    #[test]
    fn cache_child_process() {
        let Some(root) = std::env::var_os("RZ_RUNTIME_CACHE_TEST_ROOT") else {
            return;
        };
        let root = PathBuf::from(root);
        let child = std::env::var("RZ_RUNTIME_CACHE_TEST_CHILD").unwrap();
        assert!(matches!(child.as_str(), "0" | "1"));
        let cache = RuntimeCache::open(&root.join("cache")).unwrap();
        let pin = cache
            .library(&root.join("library.fixture"), &asset::hex_sha256(PAYLOAD))
            .unwrap();
        fs::write(
            root.join(format!("child-{child}.json")),
            serde_json::to_vec(&pin.storage()).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn concurrent_processes_share_one_atomically_published_entry() {
        let fixture = Fixture::new();
        let mut children: Vec<_> = (0..2)
            .map(|child| {
                Command::new(std::env::current_exe().unwrap())
                    .args([
                        "--exact",
                        "runtime_pin::cache::tests::cache_child_process",
                        "--nocapture",
                    ])
                    // E clears the runner/engine environment. Explicit cache
                    // handoff must also work without HOME/XDG/APPDATA.
                    .env_clear()
                    .env("RZ_RUNTIME_CACHE_TEST_ROOT", &fixture.root)
                    .env("RZ_RUNTIME_CACHE_TEST_CHILD", child.to_string())
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .spawn()
                    .unwrap()
            })
            .collect();
        let until = Instant::now() + Duration::from_secs(10);
        for child in &mut children {
            loop {
                if let Some(status) = child.try_wait().unwrap() {
                    assert!(status.success());
                    break;
                }
                if Instant::now() >= until {
                    child.kill().unwrap();
                    child.wait().unwrap();
                    panic!("finite child cache test timed out");
                }
                thread::sleep(Duration::from_millis(10));
            }
        }
        let receipts: Vec<serde_json::Value> = (0..2)
            .map(|child| {
                serde_json::from_slice(
                    &fs::read(fixture.root.join(format!("child-{child}.json"))).unwrap(),
                )
                .unwrap()
            })
            .collect();
        assert_eq!(
            receipts
                .iter()
                .filter(|receipt| receipt["origin"] == "cache_created")
                .count(),
            1
        );
        assert_eq!(
            receipts
                .iter()
                .filter(|receipt| receipt["origin"] == "cache_reused")
                .count(),
            1
        );
        assert_eq!(fs::read_dir(&fixture.cache.root).unwrap().count(), 1);
    }

    #[cfg(unix)]
    #[test]
    fn symlink_roots_and_replaced_cache_files_are_rejected() {
        use std::os::unix::fs::{symlink, PermissionsExt};
        let fixture = Fixture::new();
        let alias = fixture.root.join("alias");
        symlink(&fixture.cache.root, &alias).unwrap();
        assert!(RuntimeCache::open(&alias.join("nested")).is_err());
        let pin = fixture.load();
        let library = pin.path().to_owned();
        let entry = library.parent().unwrap().to_owned();
        drop(pin);
        fs::set_permissions(&entry, fs::Permissions::from_mode(0o700)).unwrap();
        fs::remove_file(&library).unwrap();
        symlink(&fixture.source, &library).unwrap();
        fs::set_permissions(&entry, fs::Permissions::from_mode(0o500)).unwrap();
        assert_eq!(
            fixture
                .cache
                .library(&fixture.source, &asset::hex_sha256(PAYLOAD))
                .unwrap_err()
                .kind,
            K::IdentityMismatch
        );
    }
}
