//! Lifetime of one utility IPC pathname, separate from tensor/device fences.
//!
//! The original control directory is private0700 and exclusively written by
//! this utility owner. A same-UID/Root writer racing the final name is outside
//! that admitted OS scope: Linux has no compare-inode-and-unlink operation.
//! Changed/unknown identities are retained; no directory tree is removed.
#![cfg(target_os = "linux")]

use crate::ArenaError;
use crate::pals_collect_v3::training_process::{directory_file, open_directory};
use nix::{
    errno::Errno,
    fcntl::{AtFlags, OFlag, openat},
    sys::stat::{FileStat, Mode, SFlag, fstat, fstatat},
    unistd::{UnlinkatFlags, geteuid, unlinkat},
};
use std::{
    ffi::OsString,
    fs::File,
    io,
    os::{
        fd::AsRawFd,
        unix::net::{SocketAddr, UnixListener, UnixStream},
    },
    path::{Component, Path, PathBuf},
    sync::Arc,
    time::Instant,
};

fn invalid(reason: &str) -> ArenaError {
    ArenaError::Integrity(format!("Rovent utility endpoint: {reason}"))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DirectoryIdentity {
    device: nix::libc::dev_t,
    inode: nix::libc::ino_t,
    mode: nix::libc::mode_t,
    uid: nix::libc::uid_t,
    gid: nix::libc::gid_t,
}
impl DirectoryIdentity {
    fn read(file: &File) -> Result<Self, ArenaError> {
        let value = fstat(file).map_err(|_| invalid("control FD metadata unknown"))?;
        if value.st_mode & SFlag::S_IFMT.bits() != SFlag::S_IFDIR.bits()
            || value.st_mode & 0o7777 != 0o700
            || value.st_uid != geteuid().as_raw()
        {
            return Err(invalid("original private0700 control FD required"));
        }
        Ok(Self {
            device: value.st_dev,
            inode: value.st_ino,
            mode: value.st_mode,
            uid: value.st_uid,
            gid: value.st_gid,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct EndpointIdentity {
    device: nix::libc::dev_t,
    inode: nix::libc::ino_t,
    mode: nix::libc::mode_t,
    uid: nix::libc::uid_t,
    gid: nix::libc::gid_t,
    links: nix::libc::nlink_t,
    bytes: nix::libc::off_t,
}
impl EndpointIdentity {
    fn read(value: FileStat, uid: u32) -> Result<Self, ArenaError> {
        if value.st_mode & SFlag::S_IFMT.bits() != SFlag::S_IFSOCK.bits()
            || value.st_uid != uid
            || value.st_nlink != 1
            || value.st_size != 0
        {
            return Err(invalid("original one-link zero-byte owned socket required"));
        }
        Ok(Self {
            device: value.st_dev,
            inode: value.st_ino,
            mode: value.st_mode,
            uid: value.st_uid,
            gid: value.st_gid,
            links: value.st_nlink,
            bytes: value.st_size,
        })
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Cleanup {
    Armed,
    Complete,
    Preserved,
}

/// The listener closes before its last presence token is dropped. There is no
/// clone/FD escape API; accepted streams remain scoped inside serve(ctx).
pub(super) struct UtilityListener {
    listener: UnixListener,
    _presence: Arc<()>,
}
impl UtilityListener {
    pub(super) fn set_nonblocking(&self, value: bool) -> io::Result<()> {
        self.listener.set_nonblocking(value)
    }
    pub(super) fn accept(&self) -> io::Result<(UnixStream, SocketAddr)> {
        self.listener.accept()
    }
}

pub(super) struct OwnedUtilityEndpoint {
    directory: File,
    directory_path: PathBuf,
    directory_identity: DirectoryIdentity,
    name: OsString,
    presence: Arc<()>,
    until: Instant,
    attempted: bool,
    bound: bool,
    // O_PATH holds the filesystem inode, not the listener's sockfs inode. It
    // prevents inode reuse while a changed final pathname is rejected.
    inode: Option<File>,
    identity: Option<EndpointIdentity>,
    cleanup: Cleanup,
}
impl OwnedUtilityEndpoint {
    /// Prepare and place this owner in Custody before bind. Admission failure
    /// must retain it even if bind succeeds but endpoint readback fails.
    pub(super) fn prepare(
        directory: File,
        directory_path: &Path,
        socket: &Path,
        until: Instant,
    ) -> Result<Self, ArenaError> {
        if socket.parent() != Some(directory_path) || !directory_path.is_absolute() {
            return Err(invalid("one original control-directory child required"));
        }
        let name = socket
            .file_name()
            .ok_or_else(|| invalid("endpoint basename absent"))?;
        if !matches!(
            Path::new(name).components().next(),
            Some(Component::Normal(_))
        ) || Path::new(name).components().count() != 1
        {
            return Err(invalid("plain single endpoint basename required"));
        }
        let owner = Self {
            directory_identity: DirectoryIdentity::read(&directory)?,
            directory,
            directory_path: directory_path.to_path_buf(),
            name: name.to_os_string(),
            presence: Arc::new(()),
            until,
            attempted: false,
            bound: false,
            inode: None,
            identity: None,
            cleanup: Cleanup::Armed,
        };
        owner.check_directory()?;
        Ok(owner)
    }
    fn within_original_window(&self) -> Result<(), ArenaError> {
        if Instant::now() >= self.until {
            return Err(invalid(
                "original endpoint cleanup deadline expired; retained",
            ));
        }
        Ok(())
    }
    fn check_directory(&self) -> Result<(), ArenaError> {
        let current = directory_file(&open_directory(&self.directory_path)?)?;
        if DirectoryIdentity::read(&self.directory)? != self.directory_identity
            || DirectoryIdentity::read(&current)? != self.directory_identity
        {
            return Err(invalid(
                "original control FD/path changed; endpoint preserved",
            ));
        }
        Ok(())
    }
    fn named_identity(&self) -> Result<EndpointIdentity, ArenaError> {
        EndpointIdentity::read(
            fstatat(
                &self.directory,
                self.name.as_os_str(),
                AtFlags::AT_SYMLINK_NOFOLLOW,
            )
            .map_err(|_| invalid("endpoint name identity unknown"))?,
            self.directory_identity.uid,
        )
    }
    pub(super) fn bind(&mut self) -> Result<UtilityListener, ArenaError> {
        if self.attempted || self.cleanup != Cleanup::Armed {
            return Err(invalid("endpoint bind cannot retry"));
        }
        self.attempted = true;
        self.check_directory()?;
        match fstatat(
            &self.directory,
            self.name.as_os_str(),
            AtFlags::AT_SYMLINK_NOFOLLOW,
        ) {
            Err(Errno::ENOENT) => (),
            _ => return Err(invalid("endpoint must be absent before original bind")),
        }
        // Linux has no bindat. Keep the directory FD live through the procfd
        // bind, then revalidate the registered path before publishing startup.
        let anchored =
            PathBuf::from(format!("/proc/self/fd/{}", self.directory.as_raw_fd())).join(&self.name);
        if anchored.as_os_str().len() >= 100 {
            return Err(invalid("anchored endpoint path bound"));
        }
        self.within_original_window()?;
        let listener =
            UnixListener::bind(&anchored).map_err(|_| invalid("original socket bind"))?;
        self.bound = true;
        let inode = File::from(
            openat(
                &self.directory,
                self.name.as_os_str(),
                OFlag::O_PATH | OFlag::O_NOFOLLOW | OFlag::O_CLOEXEC,
                Mode::empty(),
            )
            .map_err(|_| invalid("bound endpoint identity unavailable; retained"))?,
        );
        let identity = EndpointIdentity::read(
            fstat(&inode).map_err(|_| invalid("bound endpoint FD metadata unknown"))?,
            self.directory_identity.uid,
        )?;
        self.inode = Some(inode);
        self.identity = Some(identity);
        self.check_directory()?;
        if self.named_identity()? != identity {
            return Err(invalid("endpoint changed after original bind; retained"));
        }
        Ok(UtilityListener {
            listener,
            _presence: Arc::clone(&self.presence),
        })
    }
    fn cleanup_once(&mut self) -> Result<(), ArenaError> {
        if !self.bound {
            return Ok(());
        }
        self.within_original_window()?;
        if Arc::strong_count(&self.presence) != 1 {
            return Err(invalid(
                "original listener/broker still live; endpoint retained",
            ));
        }
        self.check_directory()?;
        let identity = self
            .identity
            .ok_or_else(|| invalid("original bound identity unknown; retained"))?;
        let inode = self
            .inode
            .as_ref()
            .ok_or_else(|| invalid("original endpoint FD missing; retained"))?;
        if EndpointIdentity::read(
            fstat(inode).map_err(|_| invalid("held endpoint metadata unknown"))?,
            self.directory_identity.uid,
        )? != identity
            || self.named_identity()? != identity
        {
            return Err(invalid("endpoint was replaced; current entry retained"));
        }
        // This bounded check/unlink relies on the original private-directory
        // no-external-writer contract. It is not atomic compare-and-unlink.
        self.within_original_window()?;
        unlinkat(
            &self.directory,
            self.name.as_os_str(),
            UnlinkatFlags::NoRemoveDir,
        )
        .map_err(|_| invalid("original endpoint unlink failed"))?;
        self.directory
            .sync_all()
            .map_err(|_| invalid("endpoint directory fsync failed"))?;
        self.within_original_window()?;
        self.check_directory()?;
        match fstatat(
            &self.directory,
            self.name.as_os_str(),
            AtFlags::AT_SYMLINK_NOFOLLOW,
        ) {
            Err(Errno::ENOENT) => Ok(()),
            _ => Err(invalid(
                "endpoint absent readback unknown; no further deletion",
            )),
        }
    }
    /// Called after the actual broker join; a failure is sticky and never
    /// retried by Drop or replaced with a successful normal-close outcome.
    pub(super) fn cleanup(&mut self) -> Result<(), ArenaError> {
        match self.cleanup {
            Cleanup::Complete => return Ok(()),
            Cleanup::Preserved => {
                return Err(invalid("prior cleanup failed; original custody retained"));
            }
            Cleanup::Armed => (),
        }
        let result = self.cleanup_once();
        self.cleanup = if result.is_ok() {
            Cleanup::Complete
        } else {
            Cleanup::Preserved
        };
        result
    }
}
impl Drop for OwnedUtilityEndpoint {
    fn drop(&mut self) {
        if self.cleanup == Cleanup::Armed {
            // Only the explicit normal-close call can report success. Drop is
            // for early returns/unwind and cannot issue a physical receipt.
            let _ = self.cleanup();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        os::unix::fs::{DirBuilderExt, symlink},
        sync::atomic::{AtomicU64, Ordering},
        thread,
    };
    static NEXT: AtomicU64 = AtomicU64::new(1);
    struct Root(PathBuf);
    impl Root {
        fn new() -> Self {
            let base = std::env::temp_dir().join(format!(
                "rz-utility-endpoint-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::DirBuilder::new().mode(0o700).create(&base).unwrap();
            fs::DirBuilder::new()
                .mode(0o700)
                .create(base.join("control"))
                .unwrap();
            fs::write(base.join("control/other.txt"), b"original evidence").unwrap();
            Self(base)
        }
        fn control(&self) -> PathBuf {
            self.0.join("control")
        }
        fn path(&self) -> PathBuf {
            self.control().join("utility-1-1.sock")
        }
        fn bind(&self) -> (OwnedUtilityEndpoint, UtilityListener) {
            let directory = directory_file(&open_directory(&self.control()).unwrap()).unwrap();
            let until = Instant::now() + std::time::Duration::from_secs(30);
            let mut endpoint =
                OwnedUtilityEndpoint::prepare(directory, &self.control(), &self.path(), until)
                    .unwrap();
            let listener = endpoint.bind().unwrap();
            (endpoint, listener)
        }
    }
    fn anchored_endpoint_path(endpoint: &OwnedUtilityEndpoint) -> PathBuf {
        // Keep the original directory owner alive while the OS resolves this short address.
        PathBuf::from(format!("/proc/self/fd/{}", endpoint.directory.as_raw_fd()))
            .join(&endpoint.name)
    }

    impl Drop for Root {
        fn drop(&mut self) {
            for directory in [self.control(), self.0.join("moved")] {
                for name in ["utility-1-1.sock", "other.txt"] {
                    let _ = fs::remove_file(directory.join(name));
                }
                let _ = fs::remove_dir(directory);
            }
            let _ = fs::remove_dir(&self.0);
        }
    }
    #[test]
    fn joined_listener_explicit_cleanup_removes_only_original_endpoint() {
        let root = Root::new();
        let (mut endpoint, listener) = root.bind();
        let client = UnixStream::connect(anchored_endpoint_path(&endpoint)).unwrap();
        let (accepted, _) = listener.accept().unwrap();
        drop(client);
        drop(accepted);
        drop(listener);
        endpoint.cleanup().unwrap();
        assert!(!root.path().exists());
        assert_eq!(
            fs::read(root.control().join("other.txt")).unwrap(),
            b"original evidence"
        );
        endpoint.cleanup().unwrap();
    }
    #[test]
    fn startup_failure_after_bind_drops_listener_before_endpoint_guard() {
        let root = Root::new();
        let fail = || -> Result<(), ArenaError> {
            let (_endpoint, _listener) = root.bind();
            Err(invalid("synthetic startup failure after actual bind"))
        };
        assert!(fail().is_err());
        assert!(!root.path().exists());
        assert_eq!(
            fs::read(root.control().join("other.txt")).unwrap(),
            b"original evidence"
        );
    }
    #[test]
    fn joined_broker_failure_keeps_primary_and_cleans_original_endpoint() {
        let root = Root::new();
        let (mut endpoint, listener) = root.bind();
        let primary = thread::spawn(move || {
            let _listener = listener;
            Err::<(), _>("original broker failure")
        })
        .join()
        .unwrap()
        .unwrap_err();
        endpoint.cleanup().unwrap();
        assert_eq!(primary, "original broker failure");
        assert!(!root.path().exists());
    }
    #[test]
    fn live_listener_blocks_cleanup_and_drop_does_not_retry_it() {
        let root = Root::new();
        let (mut endpoint, listener) = root.bind();
        assert!(endpoint.cleanup().is_err());
        drop(listener);
        assert!(endpoint.cleanup().is_err());
        drop(endpoint);
        assert!(root.path().exists());
    }
    #[test]
    fn expired_original_window_preserves_endpoint_and_other_evidence() {
        let root = Root::new();
        let (mut endpoint, listener) = root.bind();
        drop(listener);
        endpoint.until = Instant::now();
        assert!(endpoint.cleanup().is_err());
        drop(endpoint);
        assert!(root.path().exists());
        assert_eq!(
            fs::read(root.control().join("other.txt")).unwrap(),
            b"original evidence"
        );
    }
    #[test]
    fn replacement_regular_symlink_and_socket_are_never_deleted() {
        for kind in 0..3 {
            let root = Root::new();
            let (mut endpoint, listener) = root.bind();
            fs::remove_file(root.path()).unwrap();
            let replacement = match kind {
                0 => {
                    fs::write(root.path(), b"replacement regular").unwrap();
                    None
                }
                1 => {
                    symlink("other.txt", root.path()).unwrap();
                    None
                }
                _ => Some(UnixListener::bind(anchored_endpoint_path(&endpoint)).unwrap()),
            };
            drop(listener);
            assert!(endpoint.cleanup().is_err());
            drop(endpoint);
            assert!(fs::symlink_metadata(root.path()).is_ok());
            if kind == 0 {
                assert_eq!(fs::read(root.path()).unwrap(), b"replacement regular");
            }
            if kind == 1 {
                assert_eq!(fs::read_link(root.path()).unwrap(), Path::new("other.txt"));
            }
            assert_eq!(
                fs::read(root.control().join("other.txt")).unwrap(),
                b"original evidence"
            );
            drop(replacement);
        }
    }
    #[test]
    fn replaced_control_path_preserves_both_original_and_replacement_entries() {
        let root = Root::new();
        let (mut endpoint, listener) = root.bind();
        fs::rename(root.control(), root.0.join("moved")).unwrap();
        fs::DirBuilder::new()
            .mode(0o700)
            .create(root.control())
            .unwrap();
        fs::write(root.path(), b"foreign control entry").unwrap();
        drop(listener);
        assert!(endpoint.cleanup().is_err());
        drop(endpoint);
        assert_eq!(fs::read(root.path()).unwrap(), b"foreign control entry");
        assert!(fs::symlink_metadata(root.0.join("moved/utility-1-1.sock")).is_ok());
        assert_eq!(
            fs::read(root.0.join("moved/other.txt")).unwrap(),
            b"original evidence"
        );
    }
}
