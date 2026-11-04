//! Direct-parent PALS system-CUDA UCI proxy building blocks.
//!
//! This module is not yet an executable arena route. A proxy must retain the
//! original child, its authenticated channel and the separate physical-owner
//! evidence until shutdown is resolved. Process exit, pipe EOF, timeout and
//! a UCI `bestmove` are not GPU completion and do not authorize killing it.

mod cli_v6;
#[allow(dead_code)]
pub(crate) mod finish;
mod parent;
#[allow(dead_code)]
mod protocol;
pub mod relay;

pub(crate) fn run_guarded_cli(arguments: &[String]) -> io::Result<()> {
    cli_v6::run_guarded_cli(arguments)
}
mod runner_v6;

use rz_experiments::{ArtifactRef, NativeRuntimeSourceV6, PalsSystemStartupLockV6};
use std::{
    fs::{self, File},
    io,
    os::fd::AsRawFd,
    os::unix::{
        fs::{FileExt, FileTypeExt, MetadataExt, PermissionsExt},
        net::UnixListener,
    },
    path::{Component, Path, PathBuf},
    process::{Child, ChildStderr, ChildStdin, ChildStdout, Command, ExitStatus, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

/// Registered child inputs supplied by the arena's original pinned owner.
/// The child's own startup rechecks its mapped executable and source lock;
/// paths or this declaration alone are not evidence of a running provider.
pub struct PalsProxyLaunchV6 {
    binary: Arc<PalsProxyBinaryPinV6>,
    startup_lock_path: PathBuf,
    parent_socket_path: PathBuf,
    lock: PalsSystemStartupLockV6,
    startup_until: Instant,
    startup_cancelled: Arc<AtomicBool>,
}

/// The verified input inode is opened once and kept for all child executions
/// that share this exact startup binary. An FD pin prevents path replacement,
/// not writes through another handle; the arena must exclude inode writers.
pub struct PalsProxyBinaryPinV6 {
    executable: File,
    cwd: File,
    private_cwd: PathBuf,
    artifact: ArtifactRef,
    image_device: u64,
    image_inode: u64,
    image_mtime: (i64, i64),
    image_ctime: (i64, i64),
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
        binary: Arc<PalsProxyBinaryPinV6>,
        startup_lock_path: PathBuf,
        parent_socket_path: PathBuf,
        lock: PalsSystemStartupLockV6,
        startup_until: Instant,
        startup_cancelled: Arc<AtomicBool>,
    ) -> io::Result<Self> {
        lock.verify().map_err(io::Error::other)?;
        let now = Instant::now();
        let max_until = now
            .checked_add(std::time::Duration::from_millis(
                lock.config.native_options.startup_ms,
            ))
            .ok_or_else(|| invalid("PALS startup deadline overflow"))?;
        if startup_cancelled.load(Ordering::Acquire)
            || startup_until <= now
            || startup_until > max_until
        {
            return Err(invalid(
                "PALS original startup control is canceled, expired or exceeds the lock",
            ));
        }
        if !matches!(
            &lock.config.system_runtime_lock.runtime,
            NativeRuntimeSourceV6::SystemCuda(_)
        ) || binary.artifact.sha256 != lock.config.endpoint.binary.sha256
            || binary.artifact.bytes != lock.config.endpoint.binary.bytes
            || !controlled_path(&startup_lock_path)
            || !controlled_path(&parent_socket_path)
            || !startup_lock_path.starts_with(&binary.private_cwd)
            || parent_socket_path.parent() != Some(binary.private_cwd.as_path())
        {
            return Err(invalid("invalid PALS direct-parent launch declaration"));
        }
        Ok(Self {
            binary,
            startup_lock_path,
            parent_socket_path,
            lock,
            startup_until,
            startup_cancelled,
        })
    }

    fn check_startup_control(&self) -> io::Result<()> {
        if self.startup_cancelled.load(Ordering::Acquire) {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "PALS original startup canceled",
            ));
        }
        if Instant::now() >= self.startup_until {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "PALS original startup deadline expired",
            ));
        }
        Ok(())
    }
}

impl PalsProxyBinaryPinV6 {
    /// Open the actual snapshot via the original held private root, verify its
    /// registered length and SHA, and retain that same readonly descriptor.
    /// Cancellation and deadline are checked between reads; an individual
    /// blocked kernel I/O call is not preempted. The source artifact's path
    /// may differ from its snapshot-relative path.
    pub fn open(
        private_cwd: PathBuf,
        executable: PathBuf,
        declared: &ArtifactRef,
        until: Instant,
        cancelled: &AtomicBool,
    ) -> io::Result<Self> {
        use nix::fcntl::{OFlag, open};
        use nix::sys::stat::Mode;

        if cancelled.load(Ordering::Acquire) {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "PALS binary pin cancelled",
            ));
        }
        if Instant::now() >= until {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "PALS binary pin deadline expired",
            ));
        }
        if !controlled_path(&private_cwd)
            || !controlled_path(&executable)
            || fs::canonicalize(&private_cwd)? != private_cwd
        {
            return Err(invalid("invalid PALS private binary path"));
        }
        let relative = executable
            .strip_prefix(&private_cwd)
            .map_err(|_| invalid("PALS binary is outside the original private root"))?;
        let relative = relative
            .to_str()
            .filter(|value| !value.is_empty())
            .ok_or_else(|| invalid("invalid PALS snapshot-relative binary path"))?;
        let cwd = File::from(
            open(
                private_cwd.as_path(),
                OFlag::O_PATH | OFlag::O_DIRECTORY | OFlag::O_NOFOLLOW | OFlag::O_CLOEXEC,
                Mode::empty(),
            )
            .map_err(io::Error::other)?,
        );
        let root = cwd.metadata()?;
        if !root.is_dir()
            || root.uid() != nix::unistd::geteuid().as_raw()
            || root.permissions().mode() & 0o077 != 0
        {
            return Err(invalid("PALS original binary root is not private"));
        }
        let root_capability = cap_std::fs::Dir::from_std_file(cwd.try_clone()?);
        let artifact = ArtifactRef {
            path: relative.to_owned(),
            ..declared.clone()
        };
        let executable = crate::pals_adaptive_startup::open_system_proxy_binary_until(
            &root_capability,
            &artifact,
            until,
            cancelled,
        )
        .map_err(io::Error::other)?;
        let image = executable.metadata()?;
        let mut header = [0u8; 4];
        executable.read_exact_at(&mut header, 0)?;
        if !image.is_file() || header != *b"\x7fELF" {
            return Err(invalid("PALS child binary must be a native ELF"));
        }
        Ok(Self {
            executable,
            cwd,
            private_cwd,
            artifact,
            image_device: image.dev(),
            image_inode: image.ino(),
            image_mtime: (image.mtime(), image.mtime_nsec()),
            image_ctime: (image.ctime(), image.ctime_nsec()),
        })
    }

    fn verify_held_metadata(&self) -> io::Result<()> {
        let image = self.executable.metadata()?;
        if !image.is_file()
            || image.dev() != self.image_device
            || image.ino() != self.image_inode
            || image.len() != self.artifact.bytes
            || (image.mtime(), image.mtime_nsec()) != self.image_mtime
            || (image.ctime(), image.ctime_nsec()) != self.image_ctime
        {
            return Err(invalid("PALS held binary changed after verification"));
        }
        Ok(())
    }
}

/// Original direct child and its authenticated liveness socket. This owner
/// intentionally exposes no timeout kill or implicit GPU completion method.
#[must_use = "the original child and its physical lifetime must be retained"]
pub struct DirectPalsChildV6 {
    child: Child,
    binary: Arc<PalsProxyBinaryPinV6>,
    mapped_image: Option<File>,
    image_after_auth_verified: bool,
    listener: UnixListener,
    channel: Option<parent::AuthenticatedChildChannel>,
    socket_path: PathBuf,
    socket_device: u64,
    socket_inode: u64,
    private_cwd: PathBuf,
    cwd_device: u64,
    cwd_inode: u64,
    lock: PalsSystemStartupLockV6,
    exit: Option<ExitStatus>,
    owner_scope: Arc<()>,
    socket_retired: bool,
    startup_until: Instant,
    startup_cancelled: Arc<AtomicBool>,
    image_grant: ImageGrantState,
    finish_channel_attempted: bool,
    finish_channel_rejected: Option<parent::AuthenticatedChildChannel>,
}

// Failed and partial grants are retained by the same child owner. Even a
// failed write may have admitted native execution, so it cannot be retried or
// translated into a safe shutdown without the original physical evidence.
enum ImageGrantState {
    NotAttempted,
    Acknowledged(parent::PalsPostImageParentV6),
    Failed(parent::PalsPostImageFailureV6),
}

/// An admission observation only; none of these states proves GPU completion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PalsImageGrantObservationV6 {
    NotAttempted,
    Acknowledged,
    Failed { grant_may_have_been_sent: bool },
}

/// Opaque same-owner physical release evidence. Child exit, socket EOF,
/// timeouts and receipt files cannot create it. The original supervisor's
/// physical evidence consumer must be connected before it can be issued.
pub struct PalsPhysicalReleaseV6 {
    owner_scope: Arc<()>,
    child_pid: u32,
    process_generation: u64,
    lock_sha256: String,
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
    pub input: PalsDirectChildInputPipeV6,
    pub output: PalsDirectChildOutputPipesV6,
}

/// Only the original authenticated child owner can create this input pipe.
pub struct PalsDirectChildInputPipeV6 {
    stdin: ChildStdin,
    owner_scope: Arc<()>,
}

/// Output pipes can be drained before parent authentication; stdin remains
/// with the original child owner until independently admitted.
pub struct PalsDirectChildOutputPipesV6 {
    stdout: ChildStdout,
    stderr: ChildStderr,
    owner_scope: Arc<()>,
}

fn retire_socket_if_unchanged(path: &Path, device: u64, inode: u64) -> io::Result<()> {
    let now = fs::symlink_metadata(path)?;
    if !now.file_type().is_socket() || now.dev() != device || now.ino() != inode {
        return Err(io::Error::other("original parent socket changed; retained"));
    }
    fs::remove_file(path)
}

impl DirectPalsChildV6 {
    pub fn image_grant_observation(&self) -> PalsImageGrantObservationV6 {
        match &self.image_grant {
            ImageGrantState::NotAttempted => PalsImageGrantObservationV6::NotAttempted,
            ImageGrantState::Acknowledged(_) => PalsImageGrantObservationV6::Acknowledged,
            ImageGrantState::Failed(failure) => PalsImageGrantObservationV6::Failed {
                grant_may_have_been_sent: failure.grant_may_have_been_sent,
            },
        }
    }

    /// Bind a socket in the caller's private 0700 directory, then directly
    /// spawn the original executable with independent stdio pipes. Successful
    /// spawn does not authenticate the child or establish native readiness.
    pub fn spawn(launch: PalsProxyLaunchV6) -> io::Result<Self> {
        launch.check_startup_control()?;
        launch.binary.verify_held_metadata()?;
        let metadata = fs::symlink_metadata(&launch.binary.private_cwd)?;
        let pinned_cwd = launch.binary.cwd.metadata()?;
        if !metadata.is_dir()
            || metadata.uid() != nix::unistd::geteuid().as_raw()
            || metadata.permissions().mode() & 0o077 != 0
            || fs::canonicalize(&launch.binary.private_cwd)? != launch.binary.private_cwd
            || metadata.dev() != pinned_cwd.dev()
            || metadata.ino() != pinned_cwd.ino()
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
        let socket = fs::symlink_metadata(&launch.parent_socket_path).map_err(|error| {
            io::Error::other(format!(
                "PALS parent socket identity unavailable; preserved: {error}"
            ))
        })?;
        if !socket.file_type().is_socket() || socket.uid() != nix::unistd::geteuid().as_raw() {
            return Err(invalid("PALS parent socket owner differs; path preserved"));
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
        let mut command = Command::new(format!(
            "/proc/self/fd/{}",
            launch.binary.executable.as_raw_fd()
        ));
        command
            .current_dir(format!("/proc/self/fd/{}", launch.binary.cwd.as_raw_fd()))
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
        if let Err(error) = launch.check_startup_control() {
            let cleanup =
                retire_socket_if_unchanged(&launch.parent_socket_path, socket.dev(), socket.ino());
            return match cleanup {
                Ok(()) => Err(error),
                Err(cleanup) => Err(io::Error::other(format!(
                    "PALS startup canceled before spawn: {error}; socket retained: {cleanup}"
                ))),
            };
        }
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
            binary: Arc::clone(&launch.binary),
            mapped_image: None,
            image_after_auth_verified: false,
            listener,
            channel: None,
            socket_path: launch.parent_socket_path,
            socket_device: socket.dev(),
            socket_inode: socket.ino(),
            private_cwd: launch.binary.private_cwd.clone(),
            cwd_device: metadata.dev(),
            cwd_inode: metadata.ino(),
            lock: launch.lock,
            exit: None,
            owner_scope: Arc::new(()),
            socket_retired: false,
            startup_until: launch.startup_until,
            startup_cancelled: launch.startup_cancelled,
            image_grant: ImageGrantState::NotAttempted,
            finish_channel_attempted: false,
            finish_channel_rejected: None,
        })
    }

    /// Authenticate only the directly spawned child and the original lock.
    /// Errors leave this exact owner and child with the caller for recovery.
    pub fn authenticate(&mut self) -> io::Result<()> {
        if self.channel.is_some() {
            return Err(invalid("PALS direct child already authenticated"));
        }
        self.observe_loaded_image()?;
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
            self.startup_until,
            self.startup_cancelled.as_ref(),
        )?);
        self.reobserve_authenticated_image()
    }

    /// Recheck the same unreaped child and held image after the peer handshake.
    /// If observation fails, retain the authenticated channel and child; the
    /// caller may retry without creating a second owner. This does not prevent
    /// a later exec or constitute a child-side native-entry barrier.
    pub fn reobserve_authenticated_image(&mut self) -> io::Result<()> {
        self.image_after_auth_verified = false;
        self.channel
            .as_ref()
            .ok_or_else(|| invalid("PALS parent is not authenticated"))?
            .ensure_live(&mut self.child)?;
        self.observe_loaded_image()?;
        self.image_after_auth_verified = true;
        Ok(())
    }

    /// Pin and inspect this actual unreaped child's mapped ELF before parent
    /// admission. This is identity evidence only, not native completion.
    fn observe_loaded_image(&mut self) -> io::Result<()> {
        if self.child.try_wait()?.is_some() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "PALS child exited before image observation",
            ));
        }
        self.binary.verify_held_metadata()?;
        let image = File::open(format!("/proc/{}/exe", self.child.id()))?;
        let metadata = image.metadata()?;
        if !metadata.is_file()
            || metadata.dev() != self.binary.image_device
            || metadata.ino() != self.binary.image_inode
        {
            return Err(invalid("PALS actual child image differs from held binary"));
        }
        self.mapped_image = Some(image);
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

    /// Drain the original stdout/stderr even before the parent handshake, so
    /// native startup cannot block on a full child pipe. This is not input or
    /// permission to infer readiness from received bytes.
    pub fn take_output_pipes(&mut self) -> io::Result<PalsDirectChildOutputPipesV6> {
        if self.child.stdout.is_none() || self.child.stderr.is_none() {
            return Err(invalid(
                "PALS direct child output pipes unavailable or already transferred",
            ));
        }
        Ok(PalsDirectChildOutputPipesV6 {
            stdout: self.child.stdout.take().expect("checked stdout"),
            stderr: self.child.stderr.take().expect("checked stderr"),
            owner_scope: Arc::clone(&self.owner_scope),
        })
    }

    /// Admit native startup only after the original parent's image check and
    /// the matching child acknowledgement. On failure, the original child,
    /// control stream and image FD (if any) remain owned by `self`.
    pub fn grant_original_image(&mut self) -> io::Result<()> {
        if !matches!(self.image_grant, ImageGrantState::NotAttempted) {
            return Err(invalid("PALS image grant already attempted"));
        }
        // Both descriptors stay with this same owner; this temporary clone is
        // not exposed to another process or independent accept loop.
        let listener = self.listener.try_clone()?;
        let until = self.startup_until;
        let cancelled = Arc::clone(&self.startup_cancelled);
        match parent::grant_after_original_image(&listener, self, until, cancelled.as_ref()) {
            Ok(grant) => {
                self.image_grant = ImageGrantState::Acknowledged(grant);
                Ok(())
            }
            Err(failure) => {
                let error = io::Error::new(failure.error.kind(), failure.error.to_string());
                self.image_grant = ImageGrantState::Failed(failure);
                Err(error)
            }
        }
    }

    /// Accept one dedicated finish channel for this same direct child.
    /// Neither its handshake nor its eventual claim is a physical fence.
    /// A failed or ambiguous handshake is sticky and cannot be retried.
    fn accept_original_finish_stream(
        &mut self,
        until: Instant,
        cancelled: &AtomicBool,
    ) -> io::Result<std::os::unix::net::UnixStream> {
        if self.finish_channel_attempted {
            return Err(invalid("PALS finish channel already attempted"));
        }
        let initial = self
            .channel
            .as_ref()
            .ok_or_else(|| invalid("PALS original parent channel unavailable"))?;
        let ImageGrantState::Acknowledged(grant) = &self.image_grant else {
            return Err(invalid("PALS original image grant unavailable"));
        };
        let NativeRuntimeSourceV6::SystemCuda(runtime) =
            &self.lock.config.system_runtime_lock.runtime
        else {
            return Err(invalid("PALS direct child has no CUDA registration"));
        };
        self.finish_channel_attempted = true;
        let channel = parent::accept_and_authenticate_finish_child(
            &self.listener,
            &mut self.child,
            self.lock.config.process_generation,
            &self.lock.canonical_sha256,
            runtime.cuda_device_id,
            until,
            cancelled,
        )?;
        match channel.into_finish_stream(
            initial,
            grant,
            &mut self.child,
            self.lock.config.process_generation,
            &self.lock.canonical_sha256,
            runtime.cuda_device_id,
        ) {
            Ok(stream) => Ok(stream),
            Err((error, channel)) => {
                self.finish_channel_rejected = Some(channel);
                Err(error)
            }
        }
    }

    fn require_authenticated_parent(&mut self) -> io::Result<()> {
        if self.startup_cancelled.load(Ordering::Acquire) {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "PALS startup cancelled before input admission",
            ));
        }
        if Instant::now() >= self.startup_until {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "PALS startup expired before input admission",
            ));
        }
        if !self.image_after_auth_verified {
            return Err(invalid(
                "PALS child image was not verified after authentication",
            ));
        }
        let initial = self
            .channel
            .as_ref()
            .ok_or_else(|| invalid("PALS parent is not authenticated"))?;
        match &self.image_grant {
            ImageGrantState::Acknowledged(grant) => grant.ensure_live(initial, &mut self.child),
            ImageGrantState::NotAttempted => Err(invalid("PALS image grant is not acknowledged")),
            ImageGrantState::Failed(_) => Err(invalid("PALS image grant failed; owner retained")),
        }
    }

    /// Transfer stdin only after the original child authenticated its parent
    /// and acknowledged this owner's image grant. This is not readiness.
    pub fn take_stdin(&mut self) -> io::Result<PalsDirectChildInputPipeV6> {
        self.require_authenticated_parent()?;
        let stdin =
            self.child.stdin.take().ok_or_else(|| {
                invalid("PALS direct child stdin unavailable or already transferred")
            })?;
        Ok(PalsDirectChildInputPipeV6 {
            stdin,
            owner_scope: Arc::clone(&self.owner_scope),
        })
    }

    /// Transfer all original pipes at once after authentication. Callers that need
    /// pre-authentication output draining should use `take_output_pipes` and
    /// then `take_stdin` instead. No duplicate stdin writer is created.
    pub fn take_pipes(&mut self) -> io::Result<PalsDirectChildPipesV6> {
        self.require_authenticated_parent()?;
        if self.child.stdin.is_none() || self.child.stdout.is_none() || self.child.stderr.is_none()
        {
            return Err(invalid(
                "PALS direct child pipes unavailable or already transferred",
            ));
        }
        Ok(PalsDirectChildPipesV6 {
            input: PalsDirectChildInputPipeV6 {
                stdin: self.child.stdin.take().expect("checked stdin"),
                owner_scope: Arc::clone(&self.owner_scope),
            },
            output: PalsDirectChildOutputPipesV6 {
                stdout: self.child.stdout.take().expect("checked stdout"),
                stderr: self.child.stderr.take().expect("checked stderr"),
                owner_scope: Arc::clone(&self.owner_scope),
            },
        })
    }

    /// A caller may retire this socket only after it has separately settled
    /// the original child and physical owner. This method checks path identity
    /// but does not itself establish either condition.
    pub fn socket_identity(&self) -> (&Path, u64, u64) {
        (&self.socket_path, self.socket_device, self.socket_inode)
    }

    /// The proof must originate from the original supervisor's validated
    /// physical-release channel, bound to this exact owner. There is no such
    /// producer yet, so production code cannot retire a spawned socket until
    /// that integration exists. Rechecking a private directory and its socket
    /// limits replacement mistakes but is not an atomic inode/unlink proof.
    pub fn retire_socket(&mut self, proof: &PalsPhysicalReleaseV6) -> io::Result<()> {
        if self.socket_retired {
            return Err(invalid("PALS parent socket already retired"));
        }
        if !Arc::ptr_eq(&self.owner_scope, &proof.owner_scope)
            || proof.child_pid != self.child.id()
            || proof.process_generation != self.lock.config.process_generation
            || proof.lock_sha256 != self.lock.canonical_sha256
        {
            return Err(invalid(
                "PALS physical release does not belong to original child",
            ));
        }
        if self.observe()?.child_exit.is_none() {
            return Err(invalid("PALS child has not exited; socket preserved"));
        }
        let cwd = fs::symlink_metadata(&self.private_cwd)?;
        if !cwd.is_dir()
            || cwd.dev() != self.cwd_device
            || cwd.ino() != self.cwd_inode
            || cwd.uid() != nix::unistd::geteuid().as_raw()
            || cwd.permissions().mode() & 0o077 != 0
            || fs::canonicalize(&self.private_cwd)? != self.private_cwd
        {
            return Err(invalid("PALS private directory changed; socket preserved"));
        }
        retire_socket_if_unchanged(&self.socket_path, self.socket_device, self.socket_inode)?;
        self.socket_retired = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest as _, Sha256};
    use std::{
        os::unix::fs::DirBuilderExt,
        time::{Duration, SystemTime, UNIX_EPOCH},
    };

    struct FixtureDir(PathBuf);
    impl Drop for FixtureDir {
        fn drop(&mut self) {
            let _ = fs::remove_file(self.0.join("binary"));
            let _ = fs::remove_file(self.0.join("original"));
            let _ = fs::remove_dir(&self.0);
        }
    }

    #[test]
    fn binary_pin_keeps_original_inode_after_path_replacement() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root =
            std::env::temp_dir().join(format!("rz-proxy-pin-{}-{nonce}", std::process::id()));
        fs::DirBuilder::new().mode(0o700).create(&root).unwrap();
        let fixture = FixtureDir(fs::canonicalize(root).unwrap());
        let path = fixture.0.join("binary");
        let bytes = b"\x7fELFtest-fixture";
        fs::write(&path, bytes).unwrap();
        let artifact = ArtifactRef {
            path: "original".into(),
            sha256: format!("{:x}", Sha256::digest(bytes)),
            bytes: bytes.len() as u64,
            source: "fixture".into(),
            license: "MIT".into(),
        };
        let pin = PalsProxyBinaryPinV6::open(
            fixture.0.clone(),
            path.clone(),
            &artifact,
            Instant::now() + Duration::from_secs(5),
            &AtomicBool::new(false),
        )
        .unwrap();
        fs::rename(&path, fixture.0.join("original")).unwrap();
        fs::write(&path, b"different").unwrap();
        assert_eq!(pin.executable.metadata().unwrap().ino(), pin.image_inode);
        assert_ne!(fs::metadata(&path).unwrap().ino(), pin.image_inode);
    }

    #[test]
    fn binary_pin_checks_cancel_before_opening_a_path() {
        let artifact = ArtifactRef {
            path: "binary".into(),
            sha256: "0".repeat(64),
            bytes: 4,
            source: "fixture".into(),
            license: "MIT".into(),
        };
        let error = match PalsProxyBinaryPinV6::open(
            PathBuf::from("/unopened-private-root"),
            PathBuf::from("/unopened-private-root/binary"),
            &artifact,
            Instant::now() + Duration::from_secs(5),
            &AtomicBool::new(true),
        ) {
            Ok(_) => panic!("cancelled binary pin must not open the path"),
            Err(error) => error,
        };
        assert_eq!(error.kind(), io::ErrorKind::Interrupted);
    }
}
