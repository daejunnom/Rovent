//! Original LC0 private parent endpoint. This is not a GPU fence or a process
//! supervisor. Caller retains this owner and all accepted streams until its
//! own same-session termination evidence permits explicit endpoint cleanup.
//!
//! Directory access is through an already-owned capability. Linux has no
//! compare-inode-and-unlink operation, so cleanup additionally relies on the
//! original private directory having no competing same-UID or root writer.
//! Unknown/replaced identities and every implicit drop preserve the pathname.

use cap_std::fs::Dir;
use nix::{
    errno::Errno,
    fcntl::{AtFlags, OFlag, openat},
    sys::{
        socket::{getsockopt, sockopt::PeerCredentials},
        stat::{FileStat, Mode, SFlag, fstat, fstatat},
    },
    unistd::{UnlinkatFlags, geteuid, getppid, unlinkat},
};
use std::{
    fs::File,
    io,
    os::{
        fd::AsRawFd,
        unix::{
            ffi::OsStrExt,
            net::{UnixListener, UnixStream},
        },
    },
    path::PathBuf,
    time::{Duration, Instant},
};

const ENDPOINT_NAME: &str = "lc0-system-control.sock";

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn remaining(until: Instant) -> io::Result<Duration> {
    until
        .checked_duration_since(Instant::now())
        .filter(|value| !value.is_zero())
        .ok_or_else(|| io::Error::new(io::ErrorKind::TimedOut, "LC0 endpoint deadline expired"))
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
    fn read(directory: &File) -> io::Result<Self> {
        let stat = fstat(directory).map_err(io::Error::other)?;
        if stat.st_mode & SFlag::S_IFMT.bits() != SFlag::S_IFDIR.bits()
            || stat.st_mode & 0o7777 != 0o700
            || stat.st_uid != geteuid().as_raw()
        {
            return Err(invalid(
                "LC0 endpoint requires original user-owned private0700 directory",
            ));
        }
        Ok(Self {
            device: stat.st_dev,
            inode: stat.st_ino,
            mode: stat.st_mode,
            uid: stat.st_uid,
            gid: stat.st_gid,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SocketIdentity {
    device: nix::libc::dev_t,
    inode: nix::libc::ino_t,
    mode: nix::libc::mode_t,
    uid: nix::libc::uid_t,
    gid: nix::libc::gid_t,
    links: nix::libc::nlink_t,
    bytes: nix::libc::off_t,
}

impl SocketIdentity {
    fn read(stat: FileStat, uid: u32) -> io::Result<Self> {
        if stat.st_mode & SFlag::S_IFMT.bits() != SFlag::S_IFSOCK.bits()
            || stat.st_uid != uid
            || stat.st_nlink != 1
            || stat.st_size != 0
        {
            return Err(invalid("LC0 endpoint socket identity is invalid"));
        }
        Ok(Self {
            device: stat.st_dev,
            inode: stat.st_ino,
            mode: stat.st_mode,
            uid: stat.st_uid,
            gid: stat.st_gid,
            links: stat.st_nlink,
            bytes: stat.st_size,
        })
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum CleanupState {
    Armed,
    Complete,
    Retained,
}

/// This value must be installed in the original process custody before bind.
/// A failed bind/readback leaves it owning any socket/listener already made.
pub(super) struct Lc0ParentEndpoint {
    directory: File,
    directory_identity: DirectoryIdentity,
    parent_pid: i32,
    listener: Option<UnixListener>,
    inode: Option<File>,
    socket_identity: Option<SocketIdentity>,
    bind_attempted: bool,
    bound: bool,
    cleanup: CleanupState,
}

impl Lc0ParentEndpoint {
    /// The captured parent PID must come from the same original parent ticket,
    /// before any endpoint is published. This constructor never binds.
    pub(super) fn prepare(directory: &Dir, parent_pid: i32) -> io::Result<Self> {
        if parent_pid <= 0 || getppid().as_raw() != parent_pid {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "LC0 original parent changed",
            ));
        }
        let original = directory.try_clone()?.into_std_file();
        let directory_identity = DirectoryIdentity::read(&original)?;
        // A cap-std directory can hold an O_PATH FD; reopen the same inode as
        // a directory so later explicit cleanup can sync it.
        let directory = File::from(
            openat(
                &original,
                ".",
                OFlag::O_RDONLY | OFlag::O_DIRECTORY | OFlag::O_NOFOLLOW | OFlag::O_CLOEXEC,
                Mode::empty(),
            )
            .map_err(io::Error::other)?,
        );
        if DirectoryIdentity::read(&directory)? != directory_identity {
            return Err(invalid("LC0 directory reopen changed identity"));
        }
        Ok(Self {
            directory,
            directory_identity,
            parent_pid,
            listener: None,
            inode: None,
            socket_identity: None,
            bind_attempted: false,
            bound: false,
            cleanup: CleanupState::Armed,
        })
    }

    fn check_directory(&self) -> io::Result<()> {
        if DirectoryIdentity::read(&self.directory)? != self.directory_identity {
            return Err(invalid("LC0 original directory identity changed"));
        }
        Ok(())
    }

    fn named_identity(&self) -> io::Result<SocketIdentity> {
        let stat = fstatat(&self.directory, ENDPOINT_NAME, AtFlags::AT_SYMLINK_NOFOLLOW)
            .map_err(io::Error::other)?;
        SocketIdentity::read(stat, self.directory_identity.uid)
    }

    /// This route is meaningful only while this process retains the same FD;
    /// peer authentication is still mandatory before native entry.
    pub(super) fn route(&self) -> PathBuf {
        PathBuf::from(format!(
            "/proc/{}/fd/{}",
            std::process::id(),
            self.directory.as_raw_fd()
        ))
        .join(ENDPOINT_NAME)
    }

    pub(super) fn bind(&mut self, until: Instant) -> io::Result<()> {
        if self.bind_attempted || self.cleanup != CleanupState::Armed {
            return Err(invalid("LC0 endpoint bind cannot be retried"));
        }
        self.bind_attempted = true;
        remaining(until)?;
        self.check_directory()?;
        if getppid().as_raw() != self.parent_pid {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "LC0 original parent changed",
            ));
        }
        match fstatat(&self.directory, ENDPOINT_NAME, AtFlags::AT_SYMLINK_NOFOLLOW) {
            Err(Errno::ENOENT) => (),
            _ => return Err(invalid("LC0 endpoint name is not absent")),
        }
        let anchored = PathBuf::from(format!("/proc/self/fd/{}", self.directory.as_raw_fd()))
            .join(ENDPOINT_NAME);
        if anchored.as_os_str().as_bytes().len() >= 108 {
            return Err(invalid("LC0 endpoint bind path is too long"));
        }
        let listener = UnixListener::bind(&anchored)?;
        self.bound = true;
        self.listener = Some(listener);
        let inode = File::from(
            openat(
                &self.directory,
                ENDPOINT_NAME,
                OFlag::O_PATH | OFlag::O_NOFOLLOW | OFlag::O_CLOEXEC,
                Mode::empty(),
            )
            .map_err(io::Error::other)?,
        );
        let identity = SocketIdentity::read(
            fstat(&inode).map_err(io::Error::other)?,
            self.directory_identity.uid,
        )?;
        self.inode = Some(inode);
        self.socket_identity = Some(identity);
        self.check_directory()?;
        if self.named_identity()? != identity {
            return Err(invalid("LC0 endpoint changed after bind; retained"));
        }
        self.listener
            .as_ref()
            .expect("bound listener retained")
            .set_nonblocking(true)?;
        remaining(until)?;
        Ok(())
    }

    pub(super) fn verify_bound(&self) -> io::Result<()> {
        self.check_directory()?;
        let expected = self
            .socket_identity
            .ok_or_else(|| invalid("LC0 original socket identity unknown"))?;
        let inode = self
            .inode
            .as_ref()
            .ok_or_else(|| invalid("LC0 original socket FD missing"))?;
        if !self.bound
            || SocketIdentity::read(
                fstat(inode).map_err(io::Error::other)?,
                self.directory_identity.uid,
            )? != expected
            || self.named_identity()? != expected
        {
            return Err(invalid("LC0 original endpoint changed or unavailable"));
        }
        Ok(())
    }

    /// Return only connections from the original parent PID and effective UID.
    /// This establishes connected peer identity, not per-message identity;
    /// the caller must authenticate epoch/lock/ordinal on the stream too.
    pub(super) fn accept_parent(&self, until: Instant) -> io::Result<UnixStream> {
        self.verify_bound()?;
        let listener = self
            .listener
            .as_ref()
            .ok_or_else(|| invalid("LC0 original listener unavailable"))?;
        loop {
            let left = remaining(until)?;
            if getppid().as_raw() != self.parent_pid {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "LC0 original parent changed",
                ));
            }
            match listener.accept() {
                Ok((stream, _)) => {
                    let peer = getsockopt(&stream, PeerCredentials).map_err(io::Error::other)?;
                    if peer.pid() != self.parent_pid || peer.uid() != self.directory_identity.uid {
                        drop(stream);
                        std::thread::sleep(left.min(Duration::from_millis(10)));
                        continue;
                    }
                    self.verify_bound()?;
                    if getppid().as_raw() != self.parent_pid {
                        return Err(io::Error::new(
                            io::ErrorKind::PermissionDenied,
                            "LC0 original parent changed",
                        ));
                    }
                    remaining(until)?;
                    return Ok(stream);
                }
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    std::thread::sleep(left.min(Duration::from_millis(10)))
                }
                Err(error) => return Err(error),
            }
        }
    }

    /// Explicit endpoint-only cleanup after the caller has independently
    /// verified that no connection or same-session work needs this listener.
    /// It is not a physical GPU completion witness. Failure is sticky and
    /// never retried; if post-unlink sync/readback fails, removal is unknown.
    /// There is no implicit Drop cleanup or recursive delete.
    pub(super) fn cleanup(&mut self, until: Instant) -> io::Result<()> {
        match self.cleanup {
            CleanupState::Complete => return Ok(()),
            CleanupState::Retained => {
                return Err(invalid("LC0 endpoint cleanup previously failed"));
            }
            CleanupState::Armed => (),
        }
        let result = (|| {
            remaining(until)?;
            if !self.bound {
                return Ok(());
            }
            self.verify_bound()?;
            // Close only our original listener before touching its pathname.
            self.listener.take();
            self.verify_bound()?;
            remaining(until)?;
            unlinkat(&self.directory, ENDPOINT_NAME, UnlinkatFlags::NoRemoveDir)
                .map_err(io::Error::other)?;
            self.directory.sync_all()?;
            remaining(until)?;
            self.check_directory()?;
            match fstatat(&self.directory, ENDPOINT_NAME, AtFlags::AT_SYMLINK_NOFOLLOW) {
                Err(Errno::ENOENT) => Ok(()),
                _ => Err(invalid("LC0 endpoint absence not confirmed")),
            }
        })();
        self.cleanup = if result.is_ok() {
            CleanupState::Complete
        } else {
            CleanupState::Retained
        };
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        os::unix::fs::{DirBuilderExt, FileTypeExt},
        sync::atomic::{AtomicU64, Ordering},
    };

    static NEXT: AtomicU64 = AtomicU64::new(1);

    fn private_dir() -> (PathBuf, Dir) {
        let path = std::env::temp_dir().join(format!(
            "rz-lc0-endpoint-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::DirBuilder::new().mode(0o700).create(&path).unwrap();
        let dir = Dir::open_ambient_dir(&path, cap_std::ambient_authority()).unwrap();
        (path, dir)
    }

    #[test]
    fn implicit_drop_preserves_original_socket() {
        let (path, dir) = private_dir();
        {
            let mut endpoint = Lc0ParentEndpoint::prepare(&dir, getppid().as_raw()).unwrap();
            endpoint
                .bind(Instant::now() + Duration::from_secs(5))
                .unwrap();
            endpoint.verify_bound().unwrap();
        }
        let socket = path.join(ENDPOINT_NAME);
        assert!(
            fs::symlink_metadata(&socket)
                .unwrap()
                .file_type()
                .is_socket()
        );
        fs::remove_file(socket).unwrap();
        fs::remove_dir(path).unwrap();
    }

    #[test]
    fn replaced_socket_is_not_removed_by_explicit_cleanup() {
        let (path, dir) = private_dir();
        let mut endpoint = Lc0ParentEndpoint::prepare(&dir, getppid().as_raw()).unwrap();
        endpoint
            .bind(Instant::now() + Duration::from_secs(5))
            .unwrap();
        let socket = path.join(ENDPOINT_NAME);
        let original = path.join("original.sock");
        fs::rename(&socket, &original).unwrap();
        fs::write(&socket, b"replacement").unwrap();
        assert!(
            endpoint
                .cleanup(Instant::now() + Duration::from_secs(5))
                .is_err()
        );
        assert_eq!(fs::read(&socket).unwrap(), b"replacement");
        assert!(
            endpoint
                .cleanup(Instant::now() + Duration::from_secs(5))
                .is_err()
        );
        drop(endpoint);
        fs::remove_file(socket).unwrap();
        fs::remove_file(original).unwrap();
        fs::remove_dir(path).unwrap();
    }
}
