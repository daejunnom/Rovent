//! 검증된 최종 CUDA cache 항목의 파일 소유권과 metadata 관측.
//! native loader를 호출하지 않으며 관측값을 실행 권한으로 복원하지 않는다.

use super::*;

/// 원 cache 검증에서만 발급되는 불투명 소유자. clone은 같은 FD 소유권을
/// 연장한다. 경로·JSON·caller가 작성한 계수로 생성할 수 없다.
#[derive(Clone)]
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub struct PreparedCudaArtifactTree(Arc<PreparedInner>);

#[cfg(not(target_os = "linux"))]
struct PreparedInner;

impl std::fmt::Debug for PreparedCudaArtifactTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedCudaArtifactTree")
            .finish_non_exhaustive()
    }
}
/// 위임된 root 자체를 제외한 정확한 하위 항목 관측. bytes는 regular
/// payload와 symlink 문자열의 st_size 합이며 target payload를 중복하지 않는다.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PreparedCudaArtifactTreeObservation {
    bytes: u64,
    files: usize,
    entries: usize,
    max_file_bytes: u64,
    regular_files: usize,
    aliases: usize,
}

impl PreparedCudaArtifactTreeObservation {
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
    /// regular 파일과 검증된 alias를 포함한 비-directory 항목 수.
    pub fn files(&self) -> usize {
        self.files
    }
    pub fn directories(&self) -> usize {
        0
    }
    pub fn entries(&self) -> usize {
        self.entries
    }
    pub fn max_file_bytes(&self) -> u64 {
        self.max_file_bytes
    }
    pub fn depth(&self) -> usize {
        0
    }
    pub fn regular_files(&self) -> usize {
        self.regular_files
    }
    pub fn aliases(&self) -> usize {
        self.aliases
    }
}

impl RuntimeCache {
    /// 이전 CUDA bundle artifact tree의 신규 준비를 명시적으로 거절한다.
    /// 기존 bundle metadata 조회는 별도 parser를 통해 유지한다.
    pub fn prepare_cuda_artifact_tree(
        &self,
        source_root: &Path,
        spec: &CudaRuntimeBundleSpec,
        checkpoint: &mut dyn FnMut(u64) -> Result<(), BackendError>,
    ) -> Result<PreparedCudaArtifactTree, BackendError> {
        refuse_legacy_cuda_bundle()?;
        #[cfg(target_os = "linux")]
        {
            let pin = self.cuda_bundle_observed(source_root, spec, checkpoint)?;
            checkpoint(0)?;
            let tree = PreparedCudaArtifactTree(Arc::new(PreparedInner::new(self, pin)?));
            tree.observe()?;
            checkpoint(0)?;
            Ok(tree)
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (source_root, spec, checkpoint);
            Err(BackendError::new(
                K::BackendUnavailable,
                S::Backend,
                "CUDA artifact tree preparation currently requires Linux",
            ))
        }
    }
}

impl PreparedCudaArtifactTree {
    /// 먼저 원 parent→entry binding을 확인한다. 다른 정상 directory는 false,
    /// 원 항목의 교체·누락은 오류다. 파일 경로나 FD를 외부에 발급하지 않는다.
    pub fn matches_directory(&self, directory: &File) -> Result<bool, BackendError> {
        #[cfg(target_os = "linux")]
        {
            self.0.verify_binding()?;
            let metadata = directory.metadata().map_err(|e| {
                bundle_io_error("CUDA watch compared directory metadata failed", &e)
            })?;
            Ok(metadata.is_dir() && self.0.identity.same_inode(&metadata))
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = directory;
            Err(BackendError::new(
                K::BackendUnavailable,
                S::Backend,
                "CUDA artifact tree observation currently requires Linux",
            ))
        }
    }

    /// SHA를 다시 읽지 않는다. 원 dir/file/alias FD, 정확 roster·inode·길이·
    /// readonly mode·single-link·alias 상대 target의 metadata만 대조한다.
    pub fn observe(&self) -> Result<PreparedCudaArtifactTreeObservation, BackendError> {
        #[cfg(target_os = "linux")]
        {
            self.0.observe()
        }
        #[cfg(not(target_os = "linux"))]
        {
            Err(BackendError::new(
                K::BackendUnavailable,
                S::Backend,
                "CUDA artifact tree observation currently requires Linux",
            ))
        }
    }
}

#[cfg(target_os = "linux")]
use std::{
    collections::BTreeSet,
    os::unix::{
        fs::{MetadataExt, OpenOptionsExt},
        io::AsRawFd,
    },
};

#[cfg(target_os = "linux")]
#[derive(Clone, Copy, PartialEq, Eq)]
struct Identity {
    dev: u64,
    ino: u64,
    bytes: u64,
    mode: u32,
    nlink: u64,
    mtime: i64,
    mtime_ns: i64,
    ctime: i64,
    ctime_ns: i64,
}
#[cfg(target_os = "linux")]
impl Identity {
    fn of(metadata: &fs::Metadata) -> Self {
        Self {
            dev: metadata.dev(),
            ino: metadata.ino(),
            bytes: metadata.len(),
            mode: metadata.mode(),
            nlink: metadata.nlink(),
            mtime: metadata.mtime(),
            mtime_ns: metadata.mtime_nsec(),
            ctime: metadata.ctime(),
            ctime_ns: metadata.ctime_nsec(),
        }
    }
    fn same_inode(&self, metadata: &fs::Metadata) -> bool {
        self.dev == metadata.dev() && self.ino == metadata.ino()
    }
}

#[cfg(target_os = "linux")]
struct RetainedEntry {
    name: String,
    file: File,
    identity: Identity,
    alias_target: Option<String>,
}

#[cfg(target_os = "linux")]
struct PreparedInner {
    // 원 pin을 보유하여 생성자의 출처와 원 19개 FD를 연장한다.
    _pin: RuntimeLibraryPin,
    parent_path: PathBuf,
    parent: File,
    parent_identity: Identity,
    name: String,
    directory: File,
    identity: Identity,
    entries: Vec<RetainedEntry>,
}

#[cfg(target_os = "linux")]
fn fd_path(file: &File) -> PathBuf {
    PathBuf::from(format!("/proc/self/fd/{}", file.as_raw_fd()))
}

#[cfg(target_os = "linux")]
fn open_directory(path: &Path) -> Result<File, BackendError> {
    use std::path::Component;
    if !path.is_absolute() {
        return Err(bundle_identity_error(
            "CUDA watch directory must be absolute",
        ));
    }
    let mut directory = open_directory_at(Path::new("/"))?;
    for component in path.components() {
        match component {
            Component::RootDir => {}
            Component::Normal(name) => {
                directory = open_directory_at(&fd_path(&directory).join(name))?;
            }
            _ => {
                return Err(bundle_identity_error(
                    "CUDA watch directory contains traversal",
                ))
            }
        }
    }
    Ok(directory)
}

#[cfg(target_os = "linux")]
fn open_directory_at(path: &Path) -> Result<File, BackendError> {
    OpenOptions::new()
        .read(true)
        .custom_flags(0x10000 | 0x20000)
        .open(path)
        .map_err(|e| bundle_io_error("cannot retain CUDA cache directory", &e))
}

#[cfg(target_os = "linux")]
fn open_inode(path: &Path) -> Result<File, BackendError> {
    // Linux O_PATH|O_NOFOLLOW: symlink를 따라가지 않고 FIFO/device를 읽지 않는다.
    OpenOptions::new()
        .read(true)
        .custom_flags(0x200000 | 0x20000)
        .open(path)
        .map_err(|e| bundle_io_error("cannot retain CUDA cache entry inode", &e))
}

#[cfg(target_os = "linux")]
fn metadata(file: &File) -> Result<fs::Metadata, BackendError> {
    file.metadata()
        .map_err(|e| bundle_io_error("CUDA cache retained metadata failed", &e))
}

#[cfg(target_os = "linux")]
impl PreparedInner {
    fn new(cache: &RuntimeCache, pin: RuntimeLibraryPin) -> Result<Self, BackendError> {
        let spec = pin
            .bundle_spec()
            .ok_or_else(|| bundle_identity_error("CUDA watch requires a verified bundle"))?;
        let key = spec.digest()?;
        if pin.storage().cache_key != Some(key) {
            return Err(bundle_identity_error(
                "CUDA watch requires a final cache entry",
            ));
        }
        let entry = pin
            .path()
            .parent()
            .ok_or_else(|| bundle_identity_error("CUDA cache entry lacks its directory"))?;
        if entry.parent() != Some(cache.root()) {
            return Err(bundle_identity_error(
                "CUDA watch entry is outside its original cache root",
            ));
        }
        let name = entry
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| bundle_identity_error("CUDA watch entry basename is invalid"))?
            .to_owned();
        let parent = open_directory(cache.root())?;
        let parent_identity = Identity::of(&metadata(&parent)?);
        let directory = open_directory_at(&fd_path(&parent).join(&name))?;
        let identity = Identity::of(&metadata(&directory)?);
        let bundle = pin
            .0
            .bundle
            .as_ref()
            .expect("bundle specification was checked");
        if bundle._pins.len() + 1 != bundle.files.len() {
            return Err(bundle_identity_error(
                "CUDA watch original handle count differs",
            ));
        }
        let mut entries = Vec::with_capacity(
            bundle.files.len() + spec.loader_aliases.as_deref().unwrap_or(&[]).len(),
        );
        let mut others = bundle._pins.iter();
        for declared in &bundle.files {
            let original = if declared.role == RuntimeBundleFileRole::Core {
                &pin.0._pin
            } else {
                others.next().ok_or_else(|| {
                    bundle_identity_error("CUDA watch original file handle is absent")
                })?
            };
            let file = original
                .try_clone()
                .map_err(|e| bundle_io_error("CUDA watch original FD clone failed", &e))?;
            let before = metadata(&file)?;
            if before.len() != declared.bytes
                || !before.is_file()
                || before.nlink() != 1
                || before.mode() & 0o777 != 0o400
            {
                return Err(bundle_identity_error(
                    "CUDA watch original regular identity differs",
                ));
            }
            entries.push(RetainedEntry {
                name: declared.filename.clone(),
                file,
                identity: Identity::of(&before),
                alias_target: None,
            });
        }
        for alias in spec.loader_aliases.as_deref().unwrap_or(&[]) {
            let file = open_inode(&fd_path(&directory).join(&alias.filename))?;
            let before = metadata(&file)?;
            if !before.file_type().is_symlink()
                || before.nlink() != 1
                || before.len() != alias.target.len() as u64
            {
                return Err(bundle_identity_error("CUDA watch alias inode differs"));
            }
            entries.push(RetainedEntry {
                name: alias.filename.clone(),
                file,
                identity: Identity::of(&before),
                alias_target: Some(alias.target.clone()),
            });
        }
        Ok(Self {
            _pin: pin,
            parent_path: cache.root().to_owned(),
            parent,
            parent_identity,
            name,
            directory,
            identity,
            entries,
        })
    }

    fn verify_binding(&self) -> Result<(), BackendError> {
        let parent = metadata(&self.parent)?;
        let named_parent = open_directory(&self.parent_path)?;
        let named_parent = metadata(&named_parent)?;
        if !parent.is_dir()
            || !self.parent_identity.same_inode(&parent)
            || !self.parent_identity.same_inode(&named_parent)
            || parent.mode() & 0o077 != 0
            || named_parent.mode() & 0o077 != 0
        {
            return Err(bundle_identity_error(
                "CUDA watch original cache parent changed",
            ));
        }
        let named = open_directory_at(&fd_path(&self.parent).join(&self.name))?;
        let named = metadata(&named)?;
        let retained = metadata(&self.directory)?;
        if !retained.is_dir()
            || retained.mode() & 0o777 != 0o500
            || Identity::of(&retained) != self.identity
            || Identity::of(&named) != self.identity
        {
            return Err(bundle_identity_error(
                "CUDA watch final entry directory changed",
            ));
        }
        Ok(())
    }

    fn observe(&self) -> Result<PreparedCudaArtifactTreeObservation, BackendError> {
        self.verify_binding()?;
        let directory = fd_path(&self.directory);
        let mut names = BTreeSet::new();
        for entry in fs::read_dir(&directory)
            .map_err(|e| bundle_io_error("CUDA watch roster enumeration failed", &e))?
        {
            let entry = entry.map_err(|e| bundle_io_error("CUDA watch roster entry failed", &e))?;
            if names.len() >= self.entries.len() || !names.insert(entry.file_name()) {
                return Err(bundle_identity_error(
                    "CUDA watch roster contains an extra entry",
                ));
            }
        }
        if names.len() != self.entries.len() {
            return Err(bundle_identity_error(
                "CUDA watch roster has a missing entry",
            ));
        }
        let mut observed = PreparedCudaArtifactTreeObservation {
            bytes: 0,
            files: 0,
            entries: 0,
            max_file_bytes: 0,
            regular_files: 0,
            aliases: 0,
        };
        for entry in &self.entries {
            if !names.contains(std::ffi::OsStr::new(&entry.name)) {
                return Err(bundle_identity_error("CUDA watch roster filename differs"));
            }
            let retained = metadata(&entry.file)?;
            let named = open_inode(&directory.join(&entry.name))?;
            let named = metadata(&named)?;
            if Identity::of(&retained) != entry.identity
                || Identity::of(&named) != entry.identity
                || retained.nlink() != 1
            {
                return Err(bundle_identity_error(
                    "CUDA watch original file or alias changed",
                ));
            }
            if let Some(target) = &entry.alias_target {
                if !retained.file_type().is_symlink()
                    || fs::read_link(directory.join(&entry.name))
                        .map_err(|e| bundle_io_error("CUDA watch alias target read failed", &e))?
                        != Path::new(target)
                    || !self
                        .entries
                        .iter()
                        .any(|regular| regular.name == *target && regular.alias_target.is_none())
                {
                    return Err(bundle_identity_error("CUDA watch alias target differs"));
                }
                observed.aliases += 1;
            } else {
                if !retained.is_file() || retained.mode() & 0o777 != 0o400 {
                    return Err(bundle_identity_error(
                        "CUDA watch regular file is not readonly",
                    ));
                }
                observed.regular_files += 1;
            }
            observed.bytes = observed
                .bytes
                .checked_add(retained.len())
                .ok_or_else(|| bundle_limit_error("CUDA watch byte sum overflow"))?;
            observed.max_file_bytes = observed.max_file_bytes.max(retained.len());
            observed.files += 1;
            observed.entries += 1;
        }
        self.verify_binding()?;
        Ok(observed)
    }
}
