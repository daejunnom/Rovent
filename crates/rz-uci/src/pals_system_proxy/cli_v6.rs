//! Guarded preparation for the explicit PALS V6 proxy CLI.
//!
//! This module parses and pins actual inputs but does not start a child. The
//! caller supplies the original supervisor's monotonic window and owns the
//! returned preparation. In particular, a process-local timestamp must not
//! be relabelled as the arena's earlier start time.

use super::{
    PalsProxyBinaryPinV6, PalsProxyLaunchV6, controlled_path, relay,
    runner_v6::PalsProxySupervisorV6,
};
use crate::pals_adaptive_startup::{
    PalsSystemStartupOptionsV6, read_control_owned, relative_control_path,
};
use cap_fs_ext::DirExt;
use cap_std::fs::Dir;
use rz_experiments::PalsSystemStartupLockV6;
use std::{
    fs::{self, File},
    io,
    os::{
        fd::{AsFd, OwnedFd},
        unix::fs::{MetadataExt as _, PermissionsExt},
    },
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

fn invalid(detail: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, detail)
}

fn sensitive(path: &Path) -> bool {
    path.components().any(|part| {
        let value = part.as_os_str().to_string_lossy().to_ascii_lowercase();
        value.starts_with(".env")
            || value == ".ssh"
            || value == ".aws"
            || value == ".azure"
            || value == "auth.json"
            || value == "token.json"
            || value.starts_with("id_rsa")
            || value.starts_with("id_ed25519")
            || value.starts_with("id_ecdsa")
            || value.starts_with("id_dsa")
            || [".pem", ".key", ".p12", ".pfx"]
                .iter()
                .any(|suffix| value.ends_with(suffix))
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
            .any(|needle| value.contains(needle))
    })
}

fn validate_descriptor(fd: &OwnedFd, needs_read: bool, needs_write: bool) -> io::Result<()> {
    use nix::fcntl::{FcntlArg, FdFlag, fcntl};
    let descriptor_flags = fcntl(fd, FcntlArg::F_GETFD).map_err(io::Error::other)?;
    if !FdFlag::from_bits_truncate(descriptor_flags).contains(FdFlag::FD_CLOEXEC) {
        return Err(invalid(
            "PALS proxy descriptor must not be inherited by the child",
        ));
    }
    let access = fcntl(fd, FcntlArg::F_GETFL).map_err(io::Error::other)? & nix::libc::O_ACCMODE;
    if (needs_read && access != nix::libc::O_RDONLY && access != nix::libc::O_RDWR)
        || (needs_write && access != nix::libc::O_WRONLY && access != nix::libc::O_RDWR)
    {
        return Err(invalid("PALS proxy descriptor direction is invalid"));
    }
    Ok(())
}

/// Caller-supplied supervisor bounds. Validation checks their numeric limits;
/// this value alone does not establish their origin or arena identity.
#[derive(Clone)]
pub(super) struct OriginalProxyWindowV6 {
    /// Local request reference for conservative bounds, not the arena start.
    pub bound_reference: Instant,
    pub startup_until: Instant,
    pub run_until: Instant,
    pub cleanup_until: Instant,
    pub cancelled: Arc<AtomicBool>,
    pub cleanup_cancelled: Arc<AtomicBool>,
}

struct ProxyArgumentsV6 {
    startup: PalsSystemStartupOptionsV6,
    child_path: PathBuf,
}

fn parse_arguments(arguments: &[String]) -> io::Result<ProxyArgumentsV6> {
    let mut proxy = false;
    let mut child_path = None;
    let mut startup_arguments = Vec::new();
    for argument in arguments {
        if argument == "--pals-system-proxy=true" {
            if proxy {
                return Err(invalid("duplicate PALS proxy flag"));
            }
            proxy = true;
        } else if let Some(value) = argument.strip_prefix("--pals-system-child-binary=") {
            let path = PathBuf::from(value);
            if !controlled_path(&path) || sensitive(&path) || child_path.replace(path).is_some() {
                return Err(invalid("invalid or duplicate PALS child binary"));
            }
        } else {
            startup_arguments.push(argument.clone());
        }
    }
    if !proxy {
        return Err(invalid("explicit PALS system proxy flag required"));
    }
    let startup =
        PalsSystemStartupOptionsV6::parse(&startup_arguments).map_err(io::Error::other)?;
    if startup.preflight
        || startup.parent_socket.is_none()
        || sensitive(&startup.lock_path)
        || startup
            .parent_socket
            .as_ref()
            .is_some_and(|path| sensitive(path))
    {
        return Err(invalid(
            "proxy requires an actual parent socket and no metadata preflight",
        ));
    }
    Ok(ProxyArgumentsV6 {
        startup,
        child_path: child_path.ok_or_else(|| invalid("PALS child binary required"))?,
    })
}

/// Explicit CLI routing exists before the cross-process arena window and
/// physical-owner consumer are admitted. Parsing is safe; no child starts.
pub(super) fn run_guarded_cli(arguments: &[String]) -> io::Result<()> {
    let _ = parse_arguments(arguments)?;
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "PALS system proxy awaits original arena window and physical-owner supervision",
    ))
}

/// Caller retains all input descriptors if parsing or preparation fails.
pub(super) struct ProxyCliInputsV6 {
    pub arguments: Vec<String>,
    pub window: OriginalProxyWindowV6,
    pub incoming: Option<OwnedFd>,
    pub outgoing: Option<OwnedFd>,
    pub diagnostics: Option<OwnedFd>,
    pub max_line: usize,
}

/// Holds the exact no-follow lock FD and source bytes beside the existing
/// supervisor. A held descriptor does not prevent writes by another owner.
#[must_use = "retain the original proxy preparation and its owners"]
pub(super) struct PreparedProxyCliV6 {
    supervisor: PalsProxySupervisorV6,
    window: OriginalProxyWindowV6,
    root: Dir,
    lock_file: File,
    lock_relative: String,
    lock_bytes: Vec<u8>,
    lock_device: u64,
    lock_inode: u64,
    lock_size: u64,
    lock_mtime: (i64, i64),
    lock_ctime: (i64, i64),
}

impl ProxyCliInputsV6 {
    /// Duplicate process stdio without taking the original handles. The
    /// arena remains responsible for its exclusive pipe ownership.
    pub(super) fn from_stdio(
        arguments: Vec<String>,
        window: OriginalProxyWindowV6,
    ) -> io::Result<Self> {
        let incoming = io::stdin().as_fd().try_clone_to_owned()?;
        let outgoing = io::stdout().as_fd().try_clone_to_owned()?;
        let diagnostics = io::stderr().as_fd().try_clone_to_owned()?;
        Ok(Self {
            arguments,
            window,
            incoming: Some(incoming),
            outgoing: Some(outgoing),
            diagnostics: Some(diagnostics),
            max_line: crate::engine::EngineSettings::default()
                .parser
                .max_line_bytes,
        })
    }

    /// Parses, checks the original bounds and pins the original lock and child
    /// binary. It creates no child, UCI worker, model or GPU session.
    pub(super) fn prepare(&mut self) -> io::Result<PreparedProxyCliV6> {
        let args = parse_arguments(&self.arguments)?;
        let window = &self.window;
        let check_startup = || {
            if window.cancelled.load(Ordering::Acquire) {
                return Err(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "PALS proxy startup cancelled",
                ));
            }
            if window.cleanup_cancelled.load(Ordering::Acquire) {
                return Err(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "PALS proxy cleanup already cancelled",
                ));
            }
            if Instant::now() >= window.startup_until {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "PALS proxy startup deadline expired",
                ));
            }
            Ok(())
        };
        check_startup()?;
        let root_path = std::env::current_dir()?;
        if !controlled_path(&root_path)
            || sensitive(&root_path)
            || fs::canonicalize(&root_path)? != root_path
        {
            return Err(invalid(
                "PALS proxy cwd is not the original absolute private root",
            ));
        }
        if args
            .startup
            .parent_socket
            .as_ref()
            .and_then(|path| path.parent())
            != Some(root_path.as_path())
            || !args.child_path.starts_with(&root_path)
        {
            return Err(invalid(
                "PALS proxy child or socket is outside the original private root",
            ));
        }
        let path_meta = fs::symlink_metadata(&root_path)?;
        if !path_meta.is_dir()
            || path_meta.uid() != nix::unistd::geteuid().as_raw()
            || path_meta.permissions().mode() & 0o077 != 0
        {
            return Err(invalid("PALS proxy root is not a private owned directory"));
        }
        let root = Dir::open_ambient_dir(&root_path, cap_std::ambient_authority())?;
        let held_root = root.dir_metadata()?;
        if !held_root.is_dir()
            || cap_fs_ext::MetadataExt::dev(&held_root) != path_meta.dev()
            || cap_fs_ext::MetadataExt::ino(&held_root) != path_meta.ino()
        {
            return Err(invalid("PALS proxy root changed during pinning"));
        }
        let relative =
            relative_control_path(&args.startup.lock_path, &root_path).map_err(io::Error::other)?;
        let (lock_file, lock_bytes, metadata) =
            read_control_owned(&root, &relative).map_err(io::Error::other)?;
        check_startup()?;
        let text = std::str::from_utf8(&lock_bytes).map_err(io::Error::other)?;
        let lock = PalsSystemStartupLockV6::from_json(text).map_err(io::Error::other)?;
        if lock.canonical_sha256 != args.startup.lock_sha256
            || lock.config.native_options.startup_ms != args.startup.startup_ms
        {
            return Err(invalid(
                "PALS proxy startup lock or duration differs from its declaration",
            ));
        }
        let now = Instant::now();
        let max_startup = window
            .bound_reference
            .checked_add(Duration::from_millis(lock.config.native_options.startup_ms));
        let max_run = window
            .bound_reference
            .checked_add(Duration::from_millis(lock.config.window.wall_time_max_ms));
        let max_cleanup = window
            .run_until
            .checked_add(Duration::from_millis(lock.config.window.cleanup_max_ms));
        if window.bound_reference > now
            || window.startup_until <= now
            || max_startup.is_none_or(|bound| window.startup_until > bound)
            || max_run.is_none_or(|bound| window.run_until > bound)
            || max_cleanup.is_none_or(|bound| window.cleanup_until > bound)
            || window.startup_until > window.run_until
            || window.run_until > window.cleanup_until
        {
            return Err(invalid("PALS proxy original execution window is invalid"));
        }
        let incoming = self
            .incoming
            .as_ref()
            .ok_or_else(|| invalid("PALS proxy input unavailable"))?;
        let outgoing = self
            .outgoing
            .as_ref()
            .ok_or_else(|| invalid("PALS proxy output unavailable"))?;
        let diagnostics = self
            .diagnostics
            .as_ref()
            .ok_or_else(|| invalid("PALS proxy diagnostics unavailable"))?;
        relay::fifo(incoming)?;
        relay::fifo(outgoing)?;
        validate_descriptor(incoming, true, false)?;
        validate_descriptor(outgoing, false, true)?;
        // A non-FIFO diagnostic sink remains eligible for the relay's
        // independent discard-drain fallback; only inheritance is checked.
        validate_descriptor(diagnostics, false, false)?;
        relay::validate_line_bound(self.max_line)?;
        if !metadata.is_file() || metadata.len() != lock_bytes.len() as u64 {
            return Err(invalid("PALS original lock changed during preparation"));
        }
        let binary = Arc::new(PalsProxyBinaryPinV6::open(
            root_path,
            args.child_path,
            &lock.config.endpoint.binary,
            window.startup_until,
            window.cancelled.as_ref(),
        )?);
        let binary_root = binary.cwd.metadata()?;
        let final_lock = lock_file.metadata()?;
        if binary_root.dev() != path_meta.dev()
            || binary_root.ino() != path_meta.ino()
            || final_lock.dev() != metadata.dev()
            || final_lock.ino() != metadata.ino()
            || final_lock.len() != metadata.len()
            || (final_lock.mtime(), final_lock.mtime_nsec())
                != (metadata.mtime(), metadata.mtime_nsec())
            || (final_lock.ctime(), final_lock.ctime_nsec())
                != (metadata.ctime(), metadata.ctime_nsec())
        {
            return Err(invalid(
                "PALS original proxy root or lock changed during preparation",
            ));
        }
        check_startup()?;
        let launch = PalsProxyLaunchV6::new(
            binary,
            args.startup.lock_path,
            args.startup.parent_socket.expect("checked parent socket"),
            lock,
            window.startup_until,
            Arc::clone(&window.cancelled),
        )?;
        let supervisor = PalsProxySupervisorV6::new(
            launch,
            self.incoming.take().expect("checked input"),
            self.outgoing.take().expect("checked output"),
            self.diagnostics.take().expect("checked diagnostics"),
            self.max_line,
            window.bound_reference,
            window.run_until,
            window.cleanup_until,
            Arc::clone(&window.cancelled),
            Arc::clone(&window.cleanup_cancelled),
        );
        Ok(PreparedProxyCliV6 {
            supervisor,
            window: window.clone(),
            root,
            lock_file,
            lock_relative: relative,
            lock_bytes,
            lock_device: metadata.dev(),
            lock_inode: metadata.ino(),
            lock_size: metadata.len(),
            lock_mtime: (metadata.mtime(), metadata.mtime_nsec()),
            lock_ctime: (metadata.ctime(), metadata.ctime_nsec()),
        })
    }
}

impl PreparedProxyCliV6 {
    /// Recheck the same held inode and its original no-follow path. This is
    /// not proof that another process cannot write after this observation.
    pub(super) fn verify_original_lock(&self) -> io::Result<()> {
        let metadata = self.lock_file.metadata()?;
        let (parents, file) = self
            .lock_relative
            .rsplit_once('/')
            .ok_or_else(|| invalid("lock parent unavailable"))?;
        let mut directory = self.root.try_clone()?;
        for parent in parents.split('/') {
            directory = directory.open_dir_nofollow(parent)?;
        }
        let path_metadata = directory.symlink_metadata(file)?;
        if !metadata.is_file()
            || !path_metadata.is_file()
            || metadata.dev() != self.lock_device
            || metadata.ino() != self.lock_inode
            || cap_fs_ext::MetadataExt::dev(&path_metadata) != self.lock_device
            || cap_fs_ext::MetadataExt::ino(&path_metadata) != self.lock_inode
            || metadata.len() != self.lock_size
            || self.lock_bytes.len() as u64 != self.lock_size
            || (metadata.mtime(), metadata.mtime_nsec()) != self.lock_mtime
            || (metadata.ctime(), metadata.ctime_nsec()) != self.lock_ctime
        {
            return Err(invalid("original PALS proxy lock owner changed"));
        }
        Ok(())
    }

    /// Keep the existing execution guard until the same supervisor's child,
    /// quit/deadline handling and physical-unknown retention are validated.
    pub(super) fn start_guarded(&mut self) -> io::Result<()> {
        if self.window.cancelled.load(Ordering::Acquire)
            || self.window.cleanup_cancelled.load(Ordering::Acquire)
        {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "PALS original proxy or cleanup was cancelled",
            ));
        }
        if Instant::now() >= self.window.startup_until || Instant::now() >= self.window.run_until {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "PALS original proxy window expired before start",
            ));
        }
        self.verify_original_lock()?;
        let _ = &self.supervisor;
        Err(io::Error::other(
            "PALS V6 proxy execution is not yet admitted",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest as _, Sha256};

    fn arguments() -> Vec<String> {
        let digest = format!("{:x}", Sha256::digest(b"proxy lock declaration"));
        vec![
            "--pals-system-proxy=true".into(),
            "--pals-system-startup-lock=/tmp/private/inputs/startup.json".into(),
            format!("--pals-system-startup-lock-sha256={digest}"),
            "--pals-system-startup-ms=1000".into(),
            "--pals-system-parent-socket=/tmp/private/control.sock".into(),
            "--pals-system-child-binary=/tmp/private/inputs/pals-adaptive".into(),
        ]
    }

    #[test]
    fn explicit_proxy_arguments_reject_duplicates_and_preflight() {
        let args = arguments();
        assert!(parse_arguments(&args).is_ok());
        let mut duplicate = args.clone();
        duplicate.push("--pals-system-proxy=true".into());
        assert!(parse_arguments(&duplicate).is_err());
        let mut preflight = args;
        preflight.push("--preflight=true".into());
        assert!(parse_arguments(&preflight).is_err());
        let mut secret_path = arguments();
        *secret_path.last_mut().unwrap() =
            "--pals-system-child-binary=/tmp/private/.ssh/id_ed25519".into();
        assert!(parse_arguments(&secret_path).is_err());
    }
}
