//! Direct-parent PALS system-CUDA UCI proxy building blocks.
//!
//! This module is not yet an executable arena route. A proxy must retain the
//! original child, its authenticated channel and the separate physical-owner
//! evidence until shutdown is resolved. Process exit, pipe EOF, timeout and
//! a UCI `bestmove` are not GPU completion and do not authorize killing it.

mod parent;
#[allow(dead_code)]
mod protocol;
pub mod relay;

use rz_experiments::{NativeRuntimeSourceV6, PalsSystemStartupLockV6};
use std::{
    fs, io,
    os::unix::{
        fs::{FileTypeExt, MetadataExt, PermissionsExt},
        net::UnixListener,
    },
    path::{Component, Path, PathBuf},
    process::{Child, ChildStderr, ChildStdin, ChildStdout, Command, ExitStatus, Stdio},
    sync::atomic::AtomicBool,
    time::Instant,
};

/// Registered child inputs supplied by the arena's original pinned owner.
/// The child's own startup rechecks its mapped executable and source lock;
/// paths or this declaration alone are not evidence of a running provider.
pub struct PalsProxyLaunchV6 {
    executable: PathBuf,
    private_cwd: PathBuf,
    startup_lock_path: PathBuf,
    parent_socket_path: PathBuf,
    lock: PalsSystemStartupLockV6,
}

fn invalid(detail: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, detail)
}

fn controlled_path(path: &Path) -> bool {
    path.is_absolute()
        && path
            .to_str()
            .is_some_and(|s| !s.chars().any(char::is_control))
        && path
            .components()
            .all(|part| matches!(part, Component::RootDir | Component::Normal(_)))
}

impl PalsProxyLaunchV6 {
    /// The original arena must independently pin these paths and enforce its
    /// resource plan. This constructor does not start or validate a GPU.
    pub fn new(
        executable: PathBuf,
        private_cwd: PathBuf,
        startup_lock_path: PathBuf,
        parent_socket_path: PathBuf,
        lock: PalsSystemStartupLockV6,
    ) -> io::Result<Self> {
        lock.verify().map_err(io::Error::other)?;
        if !matches!(
            &lock.config.system_runtime_lock.runtime,
            NativeRuntimeSourceV6::SystemCuda(_)
        ) || !controlled_path(&executable)
            || !controlled_path(&private_cwd)
            || !controlled_path(&startup_lock_path)
            || !controlled_path(&parent_socket_path)
            || !startup_lock_path.starts_with(&private_cwd)
            || parent_socket_path.parent() != Some(private_cwd.as_path())
        {
            return Err(invalid("invalid PALS direct-parent launch declaration"));
        }
        Ok(Self {
            executable,
            private_cwd,
            startup_lock_path,
            parent_socket_path,
            lock,
        })
    }
}

/// Original direct child and its authenticated liveness socket. This owner
/// intentionally exposes no timeout kill or implicit GPU completion method.
#[must_use = "the original child and its physical lifetime must be retained"]
pub struct DirectPalsChildV6 {
    child: Child,
    listener: UnixListener,
    channel: Option<parent::AuthenticatedChildChannel>,
    socket_path: PathBuf,
    socket_device: u64,
    socket_inode: u64,
    lock: PalsSystemStartupLockV6,
    exit: Option<ExitStatus>,
}

/// `child_exit` is strictly a process observation. Even `Some(success)` says
/// nothing about the original session's physical GPU completion.
#[derive(Debug)]
pub struct PalsDirectChildObservationV6 {
    pub child_pid: u32,
    pub child_exit: Option<ExitStatus>,
    pub physical_completion: &'static str,
}

/// The originals are transferred once, so the relay can use exactly one stdin
/// writer. Child ownership and the admission channel remain in the executor.
pub struct PalsDirectChildPipesV6 {
    pub stdin: ChildStdin,
    pub stdout: ChildStdout,
    pub stderr: ChildStderr,
}

fn retire_socket_if_unchanged(path: &Path, device: u64, inode: u64) -> io::Result<()> {
    let now = fs::symlink_metadata(path)?;
    if !now.file_type().is_socket() || now.dev() != device || now.ino() != inode {
        return Err(io::Error::other("original parent socket changed; retained"));
    }
    fs::remove_file(path)
}

impl DirectPalsChildV6 {
    /// Bind a socket in the caller's private 0700 directory, then directly
    /// spawn the original executable with independent stdio pipes. Successful
    /// spawn does not authenticate the child or establish native readiness.
    pub fn spawn(launch: PalsProxyLaunchV6) -> io::Result<Self> {
        let metadata = fs::symlink_metadata(&launch.private_cwd)?;
        if !metadata.is_dir()
            || metadata.uid() != nix::unistd::geteuid().as_raw()
            || metadata.permissions().mode() & 0o077 != 0
            || fs::canonicalize(&launch.private_cwd)? != launch.private_cwd
        {
            return Err(invalid(
                "PALS direct-parent cwd is not an owned private directory",
            ));
        }
        let as_text = |path: &Path| {
            path.to_str()
                .map(str::to_owned)
                .ok_or_else(|| invalid("PALS launch path is not UTF-8"))
        };
        let lock_path = as_text(&launch.startup_lock_path)?;
        let socket_path = as_text(&launch.parent_socket_path)?;
        let listener = UnixListener::bind(&launch.parent_socket_path)?;
        let socket = fs::symlink_metadata(&launch.parent_socket_path)?;
        if !socket.file_type().is_socket() || socket.uid() != nix::unistd::geteuid().as_raw() {
            return Err(invalid("PALS parent socket owner differs"));
        }
        // The 0700 directory and authenticated peer protect this socket;
        // do not chmod a pathname that another process might replace.
        if let Err(error) = listener.set_nonblocking(true) {
            let cleanup =
                retire_socket_if_unchanged(&launch.parent_socket_path, socket.dev(), socket.ino());
            return match cleanup {
                Ok(()) => Err(error),
                Err(cleanup) => Err(io::Error::other(format!(
                    "parent listener setup failed: {error}; socket retained: {cleanup}"
                ))),
            };
        }
        let mut command = Command::new(&launch.executable);
        command
            .current_dir(&launch.private_cwd)
            .arg(format!("--pals-system-startup-lock={lock_path}"))
            .arg(format!(
                "--pals-system-startup-lock-sha256={}",
                launch.lock.canonical_sha256
            ))
            .arg(format!(
                "--pals-system-startup-ms={}",
                launch.lock.config.native_options.startup_ms
            ))
            .arg(format!("--pals-system-parent-socket={socket_path}"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let child = match command.spawn() {
            Ok(child) => child,
            Err(error) => {
                let cleanup = retire_socket_if_unchanged(
                    &launch.parent_socket_path,
                    socket.dev(),
                    socket.ino(),
                );
                return match cleanup {
                    Ok(()) => Err(error),
                    Err(cleanup) => Err(io::Error::other(format!(
                        "PALS child spawn failed: {error}; socket retained: {cleanup}"
                    ))),
                };
            }
        };
        Ok(Self {
            child,
            listener,
            channel: None,
            socket_path: launch.parent_socket_path,
            socket_device: socket.dev(),
            socket_inode: socket.ino(),
            lock: launch.lock,
            exit: None,
        })
    }

    /// Authenticate only the directly spawned child and the original lock.
    /// Errors leave this exact owner and child with the caller for recovery.
    pub fn authenticate(&mut self, until: Instant, cancelled: &AtomicBool) -> io::Result<()> {
        if self.channel.is_some() {
            return Err(invalid("PALS direct child already authenticated"));
        }
        let NativeRuntimeSourceV6::SystemCuda(runtime) =
            &self.lock.config.system_runtime_lock.runtime
        else {
            return Err(invalid("PALS direct child has no CUDA registration"));
        };
        self.channel = Some(parent::accept_and_authenticate_child(
            &self.listener,
            &mut self.child,
            self.lock.config.process_generation,
            &self.lock.canonical_sha256,
            runtime.cuda_device_id,
            until,
            cancelled,
        )?);
        Ok(())
    }

    /// Nonblocking process observation; the exact child handle remains held.
    /// The caller must not use a timeout, EOF or exit as a physical fence.
    pub fn observe(&mut self) -> io::Result<PalsDirectChildObservationV6> {
        if self.exit.is_none() {
            self.exit = self.child.try_wait()?;
        }
        Ok(PalsDirectChildObservationV6 {
            child_pid: self.child.id(),
            child_exit: self.exit,
            physical_completion: "unknown",
        })
    }

    /// Transfer the original pipes exactly once. The process handle is never
    /// exposed to the relay, and no additional stdin writer is created here.
    pub fn take_pipes(&mut self) -> io::Result<PalsDirectChildPipesV6> {
        if self.child.stdin.is_none() || self.child.stdout.is_none() || self.child.stderr.is_none()
        {
            return Err(invalid(
                "PALS direct child pipes unavailable or already transferred",
            ));
        }
        Ok(PalsDirectChildPipesV6 {
            stdin: self.child.stdin.take().expect("checked stdin"),
            stdout: self.child.stdout.take().expect("checked stdout"),
            stderr: self.child.stderr.take().expect("checked stderr"),
        })
    }

    /// A caller may retire this socket only after it has separately settled
    /// the original child and physical owner. This method checks path identity
    /// but does not itself establish either condition.
    pub fn socket_identity(&self) -> (&Path, u64, u64) {
        (&self.socket_path, self.socket_device, self.socket_inode)
    }
}
