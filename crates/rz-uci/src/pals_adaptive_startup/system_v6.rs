//! PALS V6 system-runtime input custody. No native runtime is created here.
//!
//! This is a separate owner from V5: it never translates a system runtime to a
//! bundle lock. The caller must verify the mapped executable, keep this owner
//! with the native backend, and prevent concurrent writes to the held inodes.

use super::{absolute_control_path, invalid, read_control, relative_control_path};
use cap_fs_ext::{DirExt, FollowSymlinks, OpenOptionsFollowExt};
use cap_std::fs::{Dir, OpenOptions};
use rz_experiments::{
    ArtifactRef, ManifestError, PalsStartupAssetPurposeV5, PalsStartupAssetV5,
    PalsSystemSourcePolicyV6, PalsSystemStartupLockV6, compiled_source_components_v5,
};
use sha2::{Digest as _, Sha256};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

/// V6 has a separate CLI domain; V5 arguments are never silently translated.
pub struct PalsSystemStartupOptionsV6 {
    pub lock_path: PathBuf,
    pub lock_sha256: String,
    pub startup_ms: u64,
    pub preflight: bool,
}
impl PalsSystemStartupOptionsV6 {
    pub fn parse(arguments: &[String]) -> Result<Self, ManifestError> {
        let (mut lock, mut sha, mut startup, mut preflight) = (None, None, None, false);
        for argument in arguments {
            if let Some(value) = argument.strip_prefix("--pals-system-startup-lock=") {
                if lock.replace(absolute_control_path(value)?).is_some() {
                    return Err(invalid("duplicate system lock"));
                }
            } else if let Some(value) = argument.strip_prefix("--pals-system-startup-lock-sha256=")
            {
                if !super::sha256(value) || sha.replace(value.to_owned()).is_some() {
                    return Err(invalid("invalid system lock digest"));
                }
            } else if let Some(value) = argument.strip_prefix("--pals-system-startup-ms=") {
                let value = value
                    .parse::<u64>()
                    .map_err(|_| invalid("invalid system startup window"))?;
                if !(1..=86_400_000).contains(&value) || startup.replace(value).is_some() {
                    return Err(invalid("invalid system startup window"));
                }
            } else if argument == "--preflight=true" && !preflight {
                preflight = true;
            } else {
                return Err(invalid("unsupported or duplicate system startup argument"));
            }
        }
        Ok(Self {
            lock_path: lock.ok_or_else(|| invalid("system lock required"))?,
            lock_sha256: sha.ok_or_else(|| invalid("system lock digest required"))?,
            startup_ms: startup.ok_or_else(|| invalid("system startup window required"))?,
            preflight,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PalsSystemSourceObservationV6 {
    Verified,
    Missing,
    Mismatch,
}

/// Distinguish original cancellation/expiry and actual I/O from identity or
/// allocation failure; no caller-supplied replacement control is accepted.
#[derive(Debug)]
pub(crate) enum PalsSystemSourceReadFailureV6 {
    Cancelled,
    Expired,
    Io(std::io::Error),
    Resource,
    Identity,
    Owner(ManifestError),
}
impl std::fmt::Display for PalsSystemSourceReadFailureV6 {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cancelled => write!(out, "original system source cancellation"),
            Self::Expired => write!(out, "original system source deadline expired"),
            Self::Io(error) => write!(out, "system source I/O: {error}"),
            Self::Resource => write!(out, "system source allocation or bound"),
            Self::Identity => write!(out, "system source identity changed"),
            Self::Owner(error) => write!(out, "system source original owner: {error}"),
        }
    }
}

/// Narrow source boundary, not an assertion about all shared runtime code.
/// Every byte here is embedded in the executable and named in the digest.
pub const PALS_SYSTEM_V6_SOURCE_COMPONENT: &str = "pals_system_v6";
pub fn compiled_pals_system_v6_source_digest() -> String {
    let files: &[(&str, &[u8])] = &[
        ("startup_owner", include_bytes!("system_v6.rs")),
        (
            "bootstrap",
            include_bytes!("../pals_adaptive_bootstrap/system_v6.rs"),
        ),
        (
            "model_sources",
            include_bytes!("../pals_native/startup_system_v6.rs"),
        ),
        (
            "startup_contract",
            include_bytes!("../../../rz-experiments/src/pals_system_startup_v6.rs"),
        ),
        (
            "system_cuda_loader",
            include_bytes!("../pals_adaptive_bootstrap/runtime_assets/system_cuda.rs"),
        ),
    ];
    let mut hash = Sha256::new();
    hash.update(b"rz-pals-system-v6-boundary-source/1");
    for (name, bytes) in files {
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name.as_bytes());
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
    }
    format!("{:x}", hash.finalize())
}

fn component_observations(
    declared: &BTreeMap<String, String>,
    compiled_v5: &BTreeMap<String, String>,
    compiled_v6: &str,
) -> (PalsSystemSourceObservationV6, PalsSystemSourceObservationV6) {
    let mut count = 0;
    let mut matches = true;
    for (name, digest) in declared {
        if name == PALS_SYSTEM_V6_SOURCE_COMPONENT {
            continue;
        }
        count += 1;
        matches &= compiled_v5.get(name) == Some(digest);
    }
    let subset = if count == 0 {
        PalsSystemSourceObservationV6::Missing
    } else if matches {
        PalsSystemSourceObservationV6::Verified
    } else {
        PalsSystemSourceObservationV6::Mismatch
    };
    let boundary = match declared.get(PALS_SYSTEM_V6_SOURCE_COMPONENT) {
        None => PalsSystemSourceObservationV6::Missing,
        Some(expected) if expected == compiled_v6 => PalsSystemSourceObservationV6::Verified,
        Some(_) => PalsSystemSourceObservationV6::Mismatch,
    };
    (subset, boundary)
}

/// Exact opened asset, retained independently of its pathname.
pub struct PinnedPalsSystemAssetV6 {
    declaration: PalsStartupAssetV5,
    file: File,
    stamp: FileStamp,
}
impl PinnedPalsSystemAssetV6 {
    pub fn declaration(&self) -> &PalsStartupAssetV5 {
        &self.declaration
    }
    /// Borrow the original read-only FD; callers must use positional reads and
    /// prevent all concurrent writes, including through other hard links.
    pub fn file(&self) -> &File {
        &self.file
    }
}

/// Only an input capability: it does not claim model, provider, or GPU readiness.
pub struct PinnedPalsSystemStartupV6 {
    lock: PalsSystemStartupLockV6,
    expected_lock_sha256: String,
    source_commit: PalsSystemSourceObservationV6,
    source_components: PalsSystemSourceObservationV6,
    compiled_components: BTreeMap<String, String>,
    system_v6_source: PalsSystemSourceObservationV6,
    mapped_executable: crate::process_receipts::ExecutableIdentityV1,
    original_until: Instant,
    original_cancelled: Arc<AtomicBool>,
    root_path: PathBuf,
    root: Dir,
    assets: Vec<PinnedPalsSystemAssetV6>,
}

impl PinnedPalsSystemStartupV6 {
    /// Parse the original V6 lock and hold each original asset FD. All hashing
    /// uses the caller's original deadline and cancellation, with no fallback
    /// or fabricated source identity. The lock is relative to the private cwd.
    pub fn open(
        lock_path: &Path,
        expected_lock_sha256: &str,
        expected_startup_ms: u64,
        private_cwd: &Path,
        started: Instant,
        upstream_until: Instant,
        cancelled: Arc<AtomicBool>,
    ) -> Result<Self, ManifestError> {
        if started > Instant::now() {
            return Err(invalid("system startup start time is in the future"));
        }
        let declared_until = started
            .checked_add(std::time::Duration::from_millis(expected_startup_ms))
            .ok_or_else(|| invalid("system startup deadline overflow"))?;
        let until = declared_until.min(upstream_until);
        check_window(until, &cancelled)?;
        if !super::sha256(expected_lock_sha256) {
            return Err(invalid("invalid original system startup lock digest"));
        }
        require_linux_fd_identity()?;
        if !private_cwd.is_absolute() {
            return Err(invalid("system startup cwd must be absolute"));
        }
        let lock_path = lock_path
            .to_str()
            .ok_or_else(|| invalid("system startup lock path is not UTF-8"))?;
        let lock_path = absolute_control_path(lock_path)?;
        let relative = relative_control_path(&lock_path, private_cwd)?;
        let root = Dir::open_ambient_dir(private_cwd, cap_std::ambient_authority())
            .map_err(|_| invalid("cannot pin system startup root"))?;
        // Bind the original root before reading even the lock. Repeat after
        // decoding so a rename during the control read cannot go unnoticed.
        verify_root_identity(&root, private_cwd)?;
        let bytes = read_control(&root, &relative)?;
        check_window(until, &cancelled)?;
        let text = std::str::from_utf8(&bytes).map_err(|_| invalid("system lock is not UTF-8"))?;
        let lock = PalsSystemStartupLockV6::from_json(text)?;
        if lock.canonical_sha256 != expected_lock_sha256 {
            return Err(invalid("system startup lock differs from original owner"));
        }
        check_startup_window(lock.config.native_options.startup_ms, expected_startup_ms)?;
        check_window(until, &cancelled)?;
        // Bind the admitted lock to the process actually running, rather than
        // trusting a path or a previously computed hash from another process.
        let current =
            std::env::current_dir().map_err(|_| invalid("system startup cwd unavailable"))?;
        verify_root_identity(&root, private_cwd)?;
        for parent in current.ancestors() {
            match std::fs::symlink_metadata(parent.join(".git")) {
                Ok(_) => return Err(invalid("system startup cwd is inside a source checkout")),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(_) => return Err(invalid("system startup checkout boundary unknown")),
            }
        }
        let binary = &lock.config.endpoint.binary;
        let mapped_executable =
            crate::process_receipts::ExecutableIdentityV1::observe_exact_with_control(
                binary.bytes,
                || !cancelled.load(Ordering::Acquire) && Instant::now() < until,
            )
            .map_err(|_| invalid("mapped executable identity unavailable"))?;
        if mapped_executable.sha256 != binary.sha256 || mapped_executable.bytes != binary.bytes {
            return Err(invalid(
                "mapped executable differs from system startup lock",
            ));
        }
        check_window(until, &cancelled)?;
        let compiled_commit = env!("RZ_PALS_COMPILED_SOURCE_COMMIT");
        let source_commit = match lock.config.endpoint.source_commit.as_deref() {
            None => PalsSystemSourceObservationV6::Missing,
            Some(_) if compiled_commit == "unregistered" => PalsSystemSourceObservationV6::Missing,
            Some(expected) if expected == compiled_commit => {
                PalsSystemSourceObservationV6::Verified
            }
            Some(_) => PalsSystemSourceObservationV6::Mismatch,
        };
        let compiled = compiled_source_components_v5()?;
        let (source_components, system_v6_source) = component_observations(
            &lock.config.source_components,
            &compiled,
            &compiled_pals_system_v6_source_digest(),
        );
        if lock.config.source_policy == PalsSystemSourcePolicyV6::Strict
            && (source_commit != PalsSystemSourceObservationV6::Verified
                || source_components != PalsSystemSourceObservationV6::Verified
                || system_v6_source != PalsSystemSourceObservationV6::Verified)
        {
            return Err(invalid("strict system startup source was not verified"));
        }
        let mut assets = Vec::new();
        assets
            .try_reserve_exact(lock.config.assets.len())
            .map_err(|_| invalid("system startup asset owner allocation failed"))?;
        for declaration in &lock.config.assets {
            check_window(until, &cancelled)?;
            let artifact = ArtifactRef {
                path: declaration.snapshot_path.clone(),
                ..declaration.artifact.clone()
            };
            let (file, stamp) = open_verified_until(&root, &artifact, until, &cancelled)?;
            assets.push(PinnedPalsSystemAssetV6 {
                declaration: declaration.clone(),
                file,
                stamp,
            });
        }
        Ok(Self {
            lock,
            expected_lock_sha256: expected_lock_sha256.to_owned(),
            source_commit,
            source_components,
            compiled_components: compiled,
            system_v6_source,
            mapped_executable,
            original_until: until,
            original_cancelled: cancelled,
            root_path: private_cwd.to_owned(),
            root,
            assets,
        })
    }

    pub fn private_cwd(&self) -> &Path {
        &self.root_path
    }
    pub fn lock(&self) -> &PalsSystemStartupLockV6 {
        &self.lock
    }
    pub fn source_commit(&self) -> PalsSystemSourceObservationV6 {
        self.source_commit
    }
    pub fn source_components(&self) -> PalsSystemSourceObservationV6 {
        self.source_components
    }
    /// Actual binary's earlier component observation, reused without hashing
    /// the same compiled source inventory again for this startup.
    pub(crate) fn compiled_component(&self, name: &str) -> Option<&str> {
        self.compiled_components.get(name).map(String::as_str)
    }
    pub fn system_v6_source(&self) -> PalsSystemSourceObservationV6 {
        self.system_v6_source
    }
    pub fn mapped_executable(&self) -> &crate::process_receipts::ExecutableIdentityV1 {
        &self.mapped_executable
    }
    /// Publish only metadata observed during this process's preflight. No
    /// provider, session, neural execution, or physical fence is implied.
    pub fn publish_preflight(&self) -> Result<(), ManifestError> {
        use std::io::Write;
        let runtime = self.runtime_directory_capability()?;
        let pid = std::process::id();
        let owner_name = format!("preflight-process-{pid}");
        runtime
            .create_dir(&owner_name)
            .map_err(|_| invalid("cannot create system preflight owner"))?;
        let owner = runtime
            .open_dir_nofollow(&owner_name)
            .map_err(|_| invalid("system preflight owner changed"))?;
        let status = |value| match value {
            PalsSystemSourceObservationV6::Verified => "verified",
            PalsSystemSourceObservationV6::Missing => "unknown",
            PalsSystemSourceObservationV6::Mismatch => "mismatch",
        };
        let record = serde_json::json!({
            "domain": "rz-pals-system-preflight-v6/1", "lock_sha256": self.expected_lock_sha256,
            "system_runtime_lock_sha256": self.lock.config.system_runtime_lock.canonical_sha256,
            "endpoint_id": self.lock.config.endpoint.id, "process_id": pid,
            "process_generation": self.lock.config.process_generation,
            "mapped_executable_sha256": self.mapped_executable.sha256,
            "mapped_executable_bytes": self.mapped_executable.bytes,
            "source_commit_observation": status(self.source_commit),
            "source_components_observation": status(self.source_components),
            "source_components_scope": "declared-v5-registry-subset",
            "system_v6_source_observation": status(self.system_v6_source),
            "system_v6_source_scope": "rz-pals-system-v6-boundary-source/1",
            "assets_held": self.assets.len(), "sessions_created": 0,
            "provider_initialized": false, "neural_execution": false,
            "physical_completion": "not_applicable_no_gpu_execution"
        });
        let raw =
            serde_json::to_vec(&record).map_err(|_| invalid("cannot encode system preflight"))?;
        let mut options = OpenOptions::new();
        options
            .write(true)
            .create_new(true)
            .follow(FollowSymlinks::No);
        let mut file = owner
            .open_with("pals-system-preflight.v6.json", &options)
            .map_err(|_| invalid("cannot create system preflight receipt"))?;
        file.write_all(&raw)
            .and_then(|_| file.sync_all())
            .map_err(|_| invalid("cannot persist system preflight"))?;
        Ok(())
    }
    /// Recheck the original declarations and held FD identity before use.
    /// This does not turn metadata into a proof that no writer exists.
    pub fn verify_unchanged(&self) -> Result<(), ManifestError> {
        check_window(self.original_until, &self.original_cancelled)?;
        verify_root_identity(&self.root, &self.root_path)?;
        self.lock.verify()?;
        if self.lock.canonical_sha256 != self.expected_lock_sha256
            || self.assets.len() != self.lock.config.assets.len()
        {
            return Err(invalid("system startup owner differs from original lock"));
        }
        for (held, declared) in self.assets.iter().zip(&self.lock.config.assets) {
            check_window(self.original_until, &self.original_cancelled)?;
            if &held.declaration != declared || file_stamp(&held.file)? != held.stamp {
                return Err(invalid("system startup asset owner changed"));
            }
        }
        Ok(())
    }
    pub fn assets(&self) -> &[PinnedPalsSystemAssetV6] {
        &self.assets
    }
    /// Read the already-held FD without mutating its shared cursor. This is
    /// bounded by the actual lock declaration and caller's original deadline.
    /// A successful read is not model, numerical, or provider acceptance.
    #[cfg(target_os = "linux")]
    pub(crate) fn read_asset_exact(
        &self,
        purpose: PalsStartupAssetPurposeV5,
        maximum: usize,
    ) -> Result<Vec<u8>, PalsSystemSourceReadFailureV6> {
        use std::os::unix::fs::FileExt;
        self.verify_source_unchanged()?;
        let held = self
            .asset(purpose)
            .map_err(PalsSystemSourceReadFailureV6::Owner)?;
        let length = usize::try_from(held.declaration.artifact.bytes)
            .map_err(|_| PalsSystemSourceReadFailureV6::Resource)?;
        if length > maximum {
            return Err(PalsSystemSourceReadFailureV6::Resource);
        }
        let mut raw = Vec::new();
        raw.try_reserve_exact(length)
            .map_err(|_| PalsSystemSourceReadFailureV6::Resource)?;
        raw.resize(length, 0);
        let mut offset = 0usize;
        let mut digest = Sha256::new();
        while offset < length {
            self.source_window()?;
            let end = offset.saturating_add(64 * 1024).min(length);
            let read = held
                .file
                .read_at(&mut raw[offset..end], offset as u64)
                .map_err(PalsSystemSourceReadFailureV6::Io)?;
            if read == 0 {
                return Err(PalsSystemSourceReadFailureV6::Identity);
            }
            digest.update(&raw[offset..offset + read]);
            offset += read;
        }
        self.source_window()?;
        if file_stamp(&held.file).map_err(PalsSystemSourceReadFailureV6::Owner)? != held.stamp
            || format!("{:x}", digest.finalize()) != held.declaration.artifact.sha256
        {
            return Err(PalsSystemSourceReadFailureV6::Identity);
        }
        self.source_window()?;
        Ok(raw)
    }

    #[cfg(target_os = "linux")]
    fn source_window(&self) -> Result<(), PalsSystemSourceReadFailureV6> {
        check_source_window(self.original_until, &self.original_cancelled)
    }

    #[cfg(target_os = "linux")]
    pub(crate) fn verify_source_unchanged(&self) -> Result<(), PalsSystemSourceReadFailureV6> {
        self.source_window()?;
        if let Err(error) = self.verify_unchanged() {
            self.source_window()?;
            return Err(PalsSystemSourceReadFailureV6::Owner(error));
        }
        self.source_window()
    }
    pub fn asset(
        &self,
        purpose: PalsStartupAssetPurposeV5,
    ) -> Result<&PinnedPalsSystemAssetV6, ManifestError> {
        let mut matches = self
            .assets
            .iter()
            .filter(|asset| asset.declaration.purpose == purpose);
        let result = matches
            .next()
            .ok_or_else(|| invalid("system startup asset missing"))?;
        if matches.next().is_some() {
            return Err(invalid("ambiguous system startup asset"));
        }
        Ok(result)
    }
    pub(crate) fn runtime_directory_capability(&self) -> Result<Dir, ManifestError> {
        self.root
            .open_dir_nofollow(&self.lock.config.runtime_directory)
            .map_err(|_| invalid("system runtime owner directory missing or linked"))
    }
}

#[cfg(target_os = "linux")]
fn check_source_window(
    until: Instant,
    cancelled: &AtomicBool,
) -> Result<(), PalsSystemSourceReadFailureV6> {
    if cancelled.load(Ordering::Acquire) {
        return Err(PalsSystemSourceReadFailureV6::Cancelled);
    }
    if Instant::now() >= until {
        return Err(PalsSystemSourceReadFailureV6::Expired);
    }
    Ok(())
}

fn check_window(until: Instant, cancelled: &AtomicBool) -> Result<(), ManifestError> {
    if cancelled.load(Ordering::Acquire) {
        return Err(invalid("system startup cancelled"));
    }
    if Instant::now() >= until {
        return Err(invalid("system startup deadline expired"));
    }
    Ok(())
}

fn check_startup_window(original_ms: u64, requested_ms: u64) -> Result<(), ManifestError> {
    if original_ms != requested_ms {
        return Err(invalid("system startup window differs from original lock"));
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn verify_root_identity(root: &Dir, expected_path: &Path) -> Result<(), ManifestError> {
    use std::os::unix::fs::MetadataExt;
    let current = std::env::current_dir().map_err(|_| invalid("system startup cwd unavailable"))?;
    if current != expected_path {
        return Err(invalid("system startup cwd differs from private owner"));
    }
    let held = root
        .try_clone()
        .map_err(|_| invalid("cannot clone system root capability"))?
        .into_std_file();
    let cwd = File::open(".").map_err(|_| invalid("cannot inspect actual system cwd"))?;
    let named =
        File::open(expected_path).map_err(|_| invalid("cannot inspect named system root"))?;
    let identity = |file: &File| -> Result<(u64, u64), ManifestError> {
        let metadata = file
            .metadata()
            .map_err(|_| invalid("system root identity unavailable"))?;
        if !metadata.is_dir() {
            return Err(invalid("system root is not a directory"));
        }
        Ok((metadata.dev(), metadata.ino()))
    };
    let original = identity(&held)?;
    if identity(&cwd)? != original || identity(&named)? != original {
        return Err(invalid("system root capability differs from actual cwd"));
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn verify_root_identity(_: &Dir, _: &Path) -> Result<(), ManifestError> {
    Err(invalid("system CUDA startup requires Linux FD identity"))
}

#[cfg(target_os = "linux")]
fn require_linux_fd_identity() -> Result<(), ManifestError> {
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn require_linux_fd_identity() -> Result<(), ManifestError> {
    Err(invalid("system CUDA startup requires Linux FD identity"))
}

/// Bounded to the exact original declared length, checking the same deadline
/// and cancellation between reads. Files cannot be considered immutable solely
/// from read-only FD ownership: the caller must prohibit concurrent writers.
fn open_verified_until(
    root: &Dir,
    artifact: &ArtifactRef,
    until: Instant,
    cancelled: &AtomicBool,
) -> Result<(File, FileStamp), ManifestError> {
    artifact.validate()?;
    let components: Vec<_> = artifact.path.split('/').collect();
    let (name, parents) = components
        .split_last()
        .ok_or_else(|| invalid("empty system asset path"))?;
    let mut directory = root
        .try_clone()
        .map_err(|_| invalid("cannot retain system asset root"))?;
    for component in parents {
        check_window(until, cancelled)?;
        directory = directory
            .open_dir_nofollow(component)
            .map_err(|_| invalid("system asset parent missing or linked"))?;
    }
    let named = directory
        .symlink_metadata(name)
        .map_err(|_| invalid("system asset missing"))?;
    if !named.is_file() {
        return Err(invalid("system asset is not a regular file"));
    }
    let mut options = OpenOptions::new();
    options.read(true).follow(FollowSymlinks::No);
    #[cfg(target_os = "linux")]
    {
        use cap_std::fs::OpenOptionsExt;
        options.custom_flags(nix::libc::O_NONBLOCK | nix::libc::O_NOFOLLOW);
    }
    let mut file = directory
        .open_with(name, &options)
        .map_err(|_| invalid("cannot open system asset"))?
        .into_std();
    let before = file_stamp(&file)?;
    if before.len != artifact.bytes {
        return Err(invalid("system asset length differs"));
    }
    let mut digest = Sha256::new();
    let mut remaining = artifact.bytes;
    let mut buffer = [0u8; 64 * 1024];
    while remaining > 0 {
        check_window(until, cancelled)?;
        let limit = usize::try_from(remaining.min(buffer.len() as u64)).unwrap_or(buffer.len());
        let count = file
            .read(&mut buffer[..limit])
            .map_err(|_| invalid("cannot read system asset"))?;
        if count == 0 {
            return Err(invalid("system asset ended early"));
        }
        remaining -= count as u64;
        digest.update(&buffer[..count]);
    }
    check_window(until, cancelled)?;
    let after = file_stamp(&file)?;
    if after != before || format!("{:x}", digest.finalize()) != artifact.sha256 {
        return Err(invalid("system asset changed or digest differs"));
    }
    file.seek(SeekFrom::Start(0))
        .map_err(|_| invalid("cannot rewind system asset"))?;
    Ok((file, after))
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct FileStamp {
    len: u64,
    modified: std::time::SystemTime,
    #[cfg(target_os = "linux")]
    device: u64,
    #[cfg(target_os = "linux")]
    inode: u64,
    #[cfg(target_os = "linux")]
    change_seconds: i64,
    #[cfg(target_os = "linux")]
    change_nanos: i64,
    #[cfg(target_os = "linux")]
    links: u64,
}

fn file_stamp(file: &File) -> Result<FileStamp, ManifestError> {
    let metadata = file
        .metadata()
        .map_err(|_| invalid("system asset metadata unavailable"))?;
    if !metadata.is_file() {
        return Err(invalid("system asset is not a regular file"));
    }
    #[cfg(target_os = "linux")]
    use std::os::unix::fs::MetadataExt;
    Ok(FileStamp {
        len: metadata.len(),
        modified: metadata
            .modified()
            .map_err(|_| invalid("system asset modification time unavailable"))?,
        #[cfg(target_os = "linux")]
        device: metadata.dev(),
        #[cfg(target_os = "linux")]
        inode: metadata.ino(),
        #[cfg(target_os = "linux")]
        change_seconds: metadata.ctime(),
        #[cfg(target_os = "linux")]
        change_nanos: metadata.ctime_nsec(),
        #[cfg(target_os = "linux")]
        links: metadata.nlink(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn v6_cli_requires_one_original_lock_digest_and_window() {
        let root = std::env::temp_dir();
        let args = vec![
            format!(
                "--pals-system-startup-lock={}",
                root.join("inputs/system-lock.json").display()
            ),
            format!(
                "--pals-system-startup-lock-sha256={:x}",
                Sha256::digest(b"original-lock")
            ),
            "--pals-system-startup-ms=120000".to_owned(),
            "--preflight=true".to_owned(),
        ];
        assert!(PalsSystemStartupOptionsV6::parse(&args).unwrap().preflight);
        for index in 0..3 {
            let mut missing = args.clone();
            missing.remove(index);
            assert!(PalsSystemStartupOptionsV6::parse(&missing).is_err());
        }
        let mut duplicate = args.clone();
        duplicate.push(args[0].clone());
        assert!(PalsSystemStartupOptionsV6::parse(&duplicate).is_err());
        let mut legacy = args.clone();
        legacy.push("--startup-config=inputs/legacy.json".to_owned());
        assert!(PalsSystemStartupOptionsV6::parse(&legacy).is_err());
    }

    #[test]
    fn original_startup_cancellation_and_deadline_are_not_reset() {
        let cancelled = AtomicBool::new(false);
        let future = Instant::now() + Duration::from_secs(5);
        assert!(check_window(future, &cancelled).is_ok());
        cancelled.store(true, Ordering::Release);
        assert!(check_window(future, &cancelled).is_err());
        cancelled.store(false, Ordering::Release);
        assert!(check_window(Instant::now(), &cancelled).is_err());
        assert!(check_startup_window(100, 100).is_ok());
        assert!(check_startup_window(100, 101).is_err());
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn source_window_reports_cancel_and_expiry_separately() {
        let cancelled = AtomicBool::new(false);
        assert!(matches!(
            check_source_window(Instant::now(), &cancelled),
            Err(PalsSystemSourceReadFailureV6::Expired)
        ));
        cancelled.store(true, Ordering::Release);
        assert!(matches!(
            check_source_window(Instant::now() + Duration::from_secs(1), &cancelled),
            Err(PalsSystemSourceReadFailureV6::Cancelled)
        ));
    }

    #[test]
    fn source_subset_does_not_verify_the_new_v6_boundary() {
        let v5_digest = format!("{:x}", Sha256::digest(b"declared-v5"));
        let v6_digest = compiled_pals_system_v6_source_digest();
        let compiled = BTreeMap::from([("coordinator".to_owned(), v5_digest.clone())]);
        let mut declared = compiled.clone();
        assert_eq!(
            component_observations(&declared, &compiled, &v6_digest),
            (
                PalsSystemSourceObservationV6::Verified,
                PalsSystemSourceObservationV6::Missing
            )
        );
        declared.insert(
            PALS_SYSTEM_V6_SOURCE_COMPONENT.to_owned(),
            v6_digest.clone(),
        );
        assert_eq!(
            component_observations(&declared, &compiled, &v6_digest),
            (
                PalsSystemSourceObservationV6::Verified,
                PalsSystemSourceObservationV6::Verified
            )
        );
        declared.insert(PALS_SYSTEM_V6_SOURCE_COMPONENT.to_owned(), v5_digest);
        assert_eq!(
            component_observations(&declared, &compiled, &v6_digest).1,
            PalsSystemSourceObservationV6::Mismatch
        );
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn held_root_is_bound_to_actual_process_cwd() {
        let cwd = std::env::current_dir().unwrap();
        let root = Dir::open_ambient_dir(&cwd, cap_std::ambient_authority()).unwrap();
        assert!(verify_root_identity(&root, &cwd).is_ok());
        assert!(verify_root_identity(&root, cwd.parent().unwrap()).is_err());
    }
}
