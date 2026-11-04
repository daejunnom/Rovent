//! Bounded V5 startup admission before model factories or worker creation.
//! Opened asset descriptors remain owned through the selected execution; source
//! identity and metadata probes never attest neural readiness or physical work.
use cap_fs_ext::{DirExt, FollowSymlinks, OpenOptionsFollowExt};
use cap_std::fs::{Dir, OpenOptions};
use rz_experiments::{
    ArtifactRef, ManifestError, PALS_PREFLIGHT_V5_DOMAIN, PalsAdaptivePreflightRecordV5,
    PalsAdaptiveStartupConfigV5, PalsInputLockV5, PalsStartupAssetPurposeV5,
    pals_private_snapshot_path_v5, verify_compiled_source_components_v5,
};
use sha2::{Digest as _, Sha256};
use std::{
    fs::File,
    io::{Read, Write},
    path::{Path, PathBuf},
};

mod search_settings;
pub use search_settings::*;
mod system_v6;
mod system_v6_execution;
#[cfg(all(target_os = "linux", feature = "onnx-cuda"))]
pub(crate) use system_v6::open_system_proxy_binary_until;
pub use system_v6::*;
pub use system_v6_execution::{
    PALS_SYSTEM_V6_EXECUTION_SOURCE_COMPONENT, compiled_pals_system_v6_execution_source_digest,
};

const CONTROL_BYTES_MAX: u64 = 4 * 1024 * 1024;
const ASSET_BYTES_MAX: u64 = 18 * 1024 * 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PalsStartupOptionsV5 {
    pub config_path: PathBuf,
    pub lock_path: PathBuf,
    pub config_sha256: String,
    pub metadata_preflight: bool,
}
impl PalsStartupOptionsV5 {
    /// This separate CLI does not consume or rewrite the legacy engine flags.
    pub fn parse(arguments: &[String]) -> Result<Self, ManifestError> {
        let mut config = None;
        let mut lock = None;
        let mut digest = None;
        let mut preflight = false;
        for argument in arguments {
            if let Some(value) = argument.strip_prefix("--startup-config=") {
                if config.replace(absolute_control_path(value)?).is_some() {
                    return Err(invalid("duplicate startup config"));
                }
            } else if let Some(value) = argument.strip_prefix("--startup-lock=") {
                if lock.replace(absolute_control_path(value)?).is_some() {
                    return Err(invalid("duplicate startup lock"));
                }
            } else if let Some(value) = argument.strip_prefix("--startup-config-sha256=") {
                if !sha256(value) || digest.replace(value.to_owned()).is_some() {
                    return Err(invalid("invalid or duplicate startup SHA-256"));
                }
            } else if argument == "--preflight=true" && !preflight {
                preflight = true;
            } else {
                return Err(invalid(
                    "unsupported or duplicate adaptive startup argument",
                ));
            }
        }
        Ok(Self {
            config_path: config.ok_or_else(|| invalid("missing startup config"))?,
            lock_path: lock.ok_or_else(|| invalid("missing startup lock"))?,
            config_sha256: digest.ok_or_else(|| invalid("missing startup config SHA-256"))?,
            metadata_preflight: preflight,
        })
    }
}

pub struct PinnedStartupAssetV5 {
    pub declaration: rz_experiments::PalsStartupAssetV5,
    /// This exact handle was streamed, hashed and rewound without reopening.
    pub file: File,
}
pub struct PinnedPalsStartupV5 {
    pub config: PalsAdaptiveStartupConfigV5,
    pub lock: PalsInputLockV5,
    pub config_sha256: String,
    pub metadata_preflight: bool,
    root: PathBuf,
    held_root: Dir,
    assets: Vec<PinnedStartupAssetV5>,
}
impl PinnedPalsStartupV5 {
    pub fn open(options: PalsStartupOptionsV5, private_cwd: &Path) -> Result<Self, ManifestError> {
        if !private_cwd.is_absolute() {
            return Err(invalid(
                "startup cwd must be the parent-owned private attempt",
            ));
        }
        let root = Dir::open_ambient_dir(private_cwd, cap_std::ambient_authority())
            .map_err(|_| invalid("cannot pin private startup cwd"))?;
        let config_name = relative_control_path(&options.config_path, private_cwd)?;
        let lock_name = relative_control_path(&options.lock_path, private_cwd)?;
        let config_bytes = read_control(&root, &config_name)?;
        if format!("{:x}", Sha256::digest(&config_bytes)) != options.config_sha256 {
            return Err(invalid("actual startup config bytes differ from their pin"));
        }
        let lock_bytes = read_control(&root, &lock_name)?;
        let lock = PalsInputLockV5::from_json(
            std::str::from_utf8(&lock_bytes).map_err(|_| invalid("lock is not UTF-8"))?,
        )?;
        let config = PalsAdaptiveStartupConfigV5::from_json(
            std::str::from_utf8(&config_bytes)
                .map_err(|_| invalid("startup config is not UTF-8"))?,
            &lock,
        )?;
        verify_compiled_source_components_v5(&config.source_components)?;
        let mut declared_total = 0u64;
        let mut assets = Vec::with_capacity(config.assets.len());
        for declaration in &config.assets {
            declared_total = declared_total
                .checked_add(declaration.artifact.bytes)
                .ok_or_else(|| invalid("startup asset byte sum overflow"))?;
            if declared_total > ASSET_BYTES_MAX {
                return Err(invalid(
                    "startup asset bytes exceed the owned staging limit",
                ));
            }
            let snapshot = ArtifactRef {
                path: declaration.snapshot_path.clone(),
                ..declaration.artifact.clone()
            };
            let file = snapshot.open_verified_at(&root, ASSET_BYTES_MAX)?;
            assets.push(PinnedStartupAssetV5 {
                declaration: declaration.clone(),
                file,
            });
        }
        Ok(Self {
            config,
            lock,
            config_sha256: options.config_sha256,
            metadata_preflight: options.metadata_preflight,
            root: private_cwd.to_owned(),
            held_root: root,
            assets,
        })
    }
    pub fn assets(&self) -> &[PinnedStartupAssetV5] {
        &self.assets
    }
    pub fn asset(
        &self,
        purpose: PalsStartupAssetPurposeV5,
    ) -> Result<&PinnedStartupAssetV5, ManifestError> {
        let mut matches = self
            .assets
            .iter()
            .filter(|a| a.declaration.purpose == purpose);
        let first = matches
            .next()
            .ok_or_else(|| invalid("required pinned asset is missing"))?;
        if matches.next().is_some() {
            return Err(invalid("pinned asset purpose is ambiguous"));
        }
        Ok(first)
    }
    pub(crate) fn private_root_capability(&self) -> Result<Dir, ManifestError> {
        self.held_root
            .try_clone()
            .map_err(|_| invalid("cannot retain private startup root"))
    }
    pub fn private_cwd(&self) -> &Path {
        &self.root
    }
    /// Only this startup's pinned role directory is granted to a profile owner.
    /// A renamed/replaced ambient parent cannot redirect the returned capability.
    pub(crate) fn runtime_directory_capability(&self) -> Result<Dir, ManifestError> {
        self.held_root
            .open_dir_nofollow(&self.config.runtime_directory)
            .map_err(|_| invalid("parent-owned runtime directory is missing or linked"))
    }
    pub fn metadata_record(&self) -> PalsAdaptivePreflightRecordV5 {
        let pid = std::process::id();
        PalsAdaptivePreflightRecordV5 {
            schema_version: 5,
            domain: PALS_PREFLIGHT_V5_DOMAIN.into(),
            lock_sha256: self.lock.canonical_sha256.clone(),
            startup_config_sha256: self.config_sha256.clone(),
            endpoint_id: self.config.endpoint.id.clone(),
            process_id: pid,
            process_run_id: format!("preflight-process-{pid}"),
            model_sessions_created: 0,
            role_requests_submitted: 0,
            cpu_jobs_submitted: 0,
            helper_processes_created: 0,
            training_executed: false,
            rules_only_stop_witness: true,
        }
    }
    /// Caller owns this process's private record directory. No inherited record
    /// is overwritten and no arbitrary output location is selected by a child.
    pub fn publish_metadata_record(&self) -> Result<(), ManifestError> {
        if !self.metadata_preflight {
            return Err(invalid(
                "metadata receipt requires the explicit preflight mode",
            ));
        }
        let record = self.metadata_record();
        let runtime = self.runtime_directory_capability()?;
        runtime
            .create_dir(&record.process_run_id)
            .map_err(|_| invalid("preflight record owner already exists or cannot be created"))?;
        let owner = runtime
            .open_dir_nofollow(&record.process_run_id)
            .map_err(|_| invalid("preflight owner directory changed"))?;
        let raw = serde_json::to_vec(&record)
            .map_err(|_| invalid("metadata receipt cannot be encoded"))?;
        let mut options = OpenOptions::new();
        options
            .write(true)
            .create_new(true)
            .follow(FollowSymlinks::No);
        let mut file = owner
            .open_with("pals-metadata-preflight.v5.json", &options)
            .map_err(|_| invalid("cannot exclusively create metadata receipt"))?;
        file.write_all(&raw)
            .and_then(|_| file.sync_all())
            .map_err(|_| invalid("metadata receipt persistence failed"))?;
        Ok(())
    }
}

fn absolute_control_path(value: &str) -> Result<PathBuf, ManifestError> {
    let path = PathBuf::from(value);
    if !path.is_absolute()
        || value.len() > 2048
        || value.chars().any(char::is_control)
        || path.components().any(|c| {
            let text = c.as_os_str().to_string_lossy().to_ascii_lowercase();
            text.starts_with(".env")
                || [
                    "credential",
                    "secret",
                    "service-account",
                    "service_account",
                    "ssh-key",
                    "api-key",
                    "api_key",
                ]
                .iter()
                .any(|word| text.contains(word))
        })
    {
        return Err(invalid("startup control path is unsafe or not absolute"));
    }
    Ok(path)
}
fn relative_control_path(path: &Path, root: &Path) -> Result<String, ManifestError> {
    let relative = path
        .strip_prefix(root)
        .map_err(|_| invalid("control file is outside the private startup cwd"))?;
    let name = relative
        .to_str()
        .ok_or_else(|| invalid("control path is not UTF-8"))?
        .replace('\\', "/");
    if !pals_private_snapshot_path_v5(&name) {
        return Err(invalid(
            "control file is outside inputs/ snapshot namespace",
        ));
    }
    Ok(name)
}
fn read_control(root: &Dir, path: &str) -> Result<Vec<u8>, ManifestError> {
    let (file, parents) = path
        .rsplit_once('/')
        .ok_or_else(|| invalid("missing control parent"))?;
    // rsplit_once gives (parent, filename); keep the scoped parent descriptor.
    let (parents, filename) = (file, parents);
    let mut directory = root
        .try_clone()
        .map_err(|_| invalid("cannot retain control root"))?;
    for parent in parents.split('/') {
        directory = directory
            .open_dir_nofollow(parent)
            .map_err(|_| invalid("control parent is unavailable or linked"))?;
    }
    if !directory
        .symlink_metadata(filename)
        .map_err(|_| invalid("control file missing"))?
        .is_file()
    {
        return Err(invalid("control input is not a regular file"));
    }
    let mut options = OpenOptions::new();
    options.read(true).follow(FollowSymlinks::No);
    #[cfg(target_os = "linux")]
    {
        use cap_std::fs::OpenOptionsExt;
        options.custom_flags(nix::libc::O_NONBLOCK | nix::libc::O_NOFOLLOW);
    }
    let file = directory
        .open_with(filename, &options)
        .map_err(|_| invalid("control file cannot be pinned"))?;
    let size = file
        .metadata()
        .map_err(|_| invalid("control metadata unavailable"))?
        .len();
    if !(1..=CONTROL_BYTES_MAX).contains(&size) {
        return Err(invalid("control input size exceeds bound"));
    }
    let mut raw = Vec::with_capacity(size as usize);
    file.take(CONTROL_BYTES_MAX + 1)
        .read_to_end(&mut raw)
        .map_err(|_| invalid("control input read failed"))?;
    if raw.len() as u64 != size {
        return Err(invalid("control input changed or exceeded bound"));
    }
    Ok(raw)
}
fn sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn invalid(message: &str) -> ManifestError {
    ManifestError::Integrity(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn arguments() -> Vec<String> {
        let root = std::env::temp_dir();
        vec![
            format!(
                "--startup-config={}",
                root.join("inputs/startup.json").display()
            ),
            format!("--startup-lock={}", root.join("inputs/lock.json").display()),
            format!("--startup-config-sha256={}", "a".repeat(64)),
        ]
    }
    #[test]
    fn new_cli_requires_complete_unique_nonsecret_snapshot_controls() {
        let valid = arguments();
        assert!(PalsStartupOptionsV5::parse(&valid).is_ok());
        for missing in 0..3 {
            let mut a = valid.clone();
            a.remove(missing);
            assert!(PalsStartupOptionsV5::parse(&a).is_err());
        }
        let mut duplicate = valid.clone();
        duplicate.push(valid[0].clone());
        assert!(PalsStartupOptionsV5::parse(&duplicate).is_err());
        let mut legacy = valid.clone();
        legacy.push("--pals-model=onnx".into());
        assert!(PalsStartupOptionsV5::parse(&legacy).is_err());
        assert!(absolute_control_path(".env").is_err());
        assert!(
            relative_control_path(
                &std::env::temp_dir().join("outside.json"),
                &std::env::temp_dir()
            )
            .is_err()
        );
    }
}
