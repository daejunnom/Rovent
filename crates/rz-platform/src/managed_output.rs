//! Linux-only inherited output descriptors and authenticated packet transport.
//! This is an OS observation boundary, not a lease or completion authority.

use nix::libc;
use nix::sys::socket::{
    getsockname, getsockopt, send, setsockopt, shutdown, sockopt, MsgFlags, Shutdown, SockType,
    UnixAddr,
};
use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::fs::MetadataExt;
use std::os::unix::net::UnixDatagram;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex, OnceLock, Weak,
};

/// Same-process inherited descriptors, retained together. The caller must
/// establish exclusive inheritance before any other thread can close/reuse
/// either numeric descriptor, and prevent other owners from changing the
/// channel's socket options for its lifetime. A successful seal alone does
/// not prove a lease.
pub struct ManagedOutputInheritedFds {
    directory: File,
    channel: UnixDatagram,
    shutdown_unknown: Arc<AtomicBool>,
    _exclusive_channel: Arc<()>,
}

struct ChannelRegistration {
    owner: Weak<()>,
    original_fd: RawFd,
    shutdown_unknown: Arc<AtomicBool>,
}
fn exclusive_channels() -> &'static Mutex<HashMap<u64, ChannelRegistration>> {
    static CHANNELS: OnceLock<Mutex<HashMap<u64, ChannelRegistration>>> = OnceLock::new();
    CHANNELS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn socket_cookie(fd: RawFd) -> io::Result<u64> {
    let mut cookie = 0_u64;
    let mut length = std::mem::size_of::<u64>() as libc::socklen_t;
    // SAFETY: the kernel writes at most the supplied u64 size into valid
    // storage; this is a read-only observation of the existing descriptor.
    if unsafe {
        libc::getsockopt(
            fd,
            libc::SOL_SOCKET,
            libc::SO_COOKIE,
            (&mut cookie as *mut u64).cast(),
            &mut length,
        )
    } != 0
    {
        return Err(io::Error::last_os_error());
    }
    if length as usize != std::mem::size_of::<u64>() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid socket cookie size",
        ));
    }
    Ok(cookie)
}

fn require_no_implicit_pidfd(fd: RawFd) -> io::Result<()> {
    let mut enabled = 0_i32;
    let mut length = std::mem::size_of::<i32>() as libc::socklen_t;
    // SAFETY: the kernel writes at most the declared i32 length to valid
    // storage. This is a read-only observation of the socket option.
    let result = unsafe {
        libc::getsockopt(
            fd,
            libc::SOL_SOCKET,
            libc::SO_PASSPIDFD,
            (&mut enabled as *mut i32).cast(),
            &mut length,
        )
    };
    if result != 0 {
        let error = io::Error::last_os_error();
        // Kernels predating SO_PASSPIDFD cannot attach its implicit pidfd.
        if error.raw_os_error() == Some(libc::ENOPROTOOPT) {
            return Ok(());
        }
        return Err(error);
    }
    if length as usize != std::mem::size_of::<i32>() || enabled != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "implicit SCM pidfd delivery is unsupported",
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ManagedOutputPeerCredentials {
    pub pid: i32,
    pub uid: u32,
    pub gid: u32,
}

pub struct ManagedOutputPacket {
    pub bytes: Vec<u8>,
    pub credentials: ManagedOutputPeerCredentials,
}

fn duplicate(fd: RawFd) -> io::Result<OwnedFd> {
    if fd < 3 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid inherited descriptor",
        ));
    }
    // SAFETY: fcntl validates the integer and creates an independent descriptor.
    let copy = unsafe { libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, 3_i32) };
    if copy < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: the successful F_DUPFD_CLOEXEC descriptor has a single owner.
    Ok(unsafe { OwnedFd::from_raw_fd(copy) })
}

fn mark_original_cloexec(fd: RawFd, copy: RawFd) -> io::Result<()> {
    let mut a = std::mem::MaybeUninit::<libc::stat>::uninit();
    let mut b = std::mem::MaybeUninit::<libc::stat>::uninit();
    // SAFETY: each pointer references a writable stat and each fd is checked
    // by the kernel. No result is read unless both calls succeed.
    if unsafe { libc::fstat(fd, a.as_mut_ptr()) } != 0
        || unsafe { libc::fstat(copy, b.as_mut_ptr()) } != 0
    {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: both fstat calls succeeded and initialized their outputs.
    let (a, b) = unsafe { (a.assume_init(), b.assume_init()) };
    if (a.st_dev, a.st_ino, a.st_mode) != (b.st_dev, b.st_ino, b.st_mode) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "original inherited descriptor changed",
        ));
    }
    // SAFETY: fcntl reads and updates only this descriptor's FD flags.
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFD) };
    if flags < 0 {
        return Err(io::Error::last_os_error());
    }
    if flags & libc::FD_CLOEXEC == 0
        && unsafe { libc::fcntl(fd, libc::F_SETFD, flags | libc::FD_CLOEXEC) } < 0
    {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

impl ManagedOutputInheritedFds {
    pub fn seal_exclusive(directory_fd: RawFd, channel_fd: RawFd) -> io::Result<Self> {
        if directory_fd == channel_fd {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "output and control descriptors are the same",
            ));
        }
        let directory_copy = duplicate(directory_fd)?;
        let channel_copy = duplicate(channel_fd)?;
        let directory = File::from(directory_copy);
        if !directory.metadata()?.is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "output descriptor is not a directory",
            ));
        }
        let channel = UnixDatagram::from(channel_copy);
        let kind = getsockopt(&channel, sockopt::SockType).map_err(io::Error::other)?;
        let domain: UnixAddr = getsockname(channel.as_raw_fd()).map_err(io::Error::other)?;
        let _ = domain;
        if kind != SockType::SeqPacket {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "control descriptor is not a Unix seqpacket socket",
            ));
        }
        require_no_implicit_pidfd(channel.as_raw_fd())?;
        let channel_identity = socket_cookie(channel.as_raw_fd())?;
        setsockopt(&channel, sockopt::PassCred, &true).map_err(io::Error::other)?;
        mark_original_cloexec(directory_fd, directory.as_raw_fd())?;
        mark_original_cloexec(channel_fd, channel.as_raw_fd())?;
        let exclusive = Arc::new(());
        let shutdown_unknown = Arc::new(AtomicBool::new(false));
        {
            let mut channels = match exclusive_channels().try_lock() {
                Ok(channels) => channels,
                Err(std::sync::TryLockError::WouldBlock) => {
                    return Err(io::Error::new(
                        io::ErrorKind::WouldBlock,
                        "managed output channel registration busy",
                    ))
                }
                Err(std::sync::TryLockError::Poisoned(_)) => {
                    return Err(io::Error::other("managed output channel registry poisoned"))
                }
            };
            // Keep a tombstone while the original inherited fd still names
            // this exact socket. A dropped wrapper must not permit stale ACKs
            // on the still-open original fd to seed a new sequence.
            channels.retain(|cookie, entry| {
                entry.shutdown_unknown.load(Ordering::Acquire)
                    || entry.owner.strong_count() != 0
                    || socket_cookie(entry.original_fd).ok() == Some(*cookie)
            });
            if channels.contains_key(&channel_identity) {
                return Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    "managed output channel already has an owner",
                ));
            }
            channels.insert(
                channel_identity,
                ChannelRegistration {
                    owner: Arc::downgrade(&exclusive),
                    original_fd: channel_fd,
                    shutdown_unknown: Arc::clone(&shutdown_unknown),
                },
            );
        }
        Ok(Self {
            directory,
            channel,
            shutdown_unknown,
            _exclusive_channel: exclusive,
        })
    }

    pub fn directory_identity(&self) -> io::Result<(u64, u64)> {
        let metadata = self.directory.metadata()?;
        if !metadata.is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "original output descriptor is not a directory",
            ));
        }
        Ok((metadata.dev(), metadata.ino()))
    }

    /// Duplicate the already-held descriptor without reopening a path. This
    /// is not an independent liveness or lease ownership observation.
    pub fn try_clone_directory(&self) -> io::Result<File> {
        self.directory.try_clone()
    }

    /// OS peer observation only. A caller must also bind and check each
    /// packet's SCM_CREDENTIALS against the original direct parent.
    pub fn peer_credentials(&self) -> io::Result<ManagedOutputPeerCredentials> {
        let cred = getsockopt(&self.channel, sockopt::PeerCredentials).map_err(io::Error::other)?;
        Ok(ManagedOutputPeerCredentials {
            pid: cred.pid(),
            uid: cred.uid(),
            gid: cred.gid(),
        })
    }

    pub fn try_send_packet(&self, bytes: &[u8]) -> io::Result<bool> {
        match send(
            self.channel.as_raw_fd(),
            bytes,
            MsgFlags::MSG_DONTWAIT | MsgFlags::MSG_NOSIGNAL,
        ) {
            Ok(sent) if sent == bytes.len() => Ok(true),
            Ok(_) => Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "partial seqpacket send",
            )),
            Err(nix::errno::Errno::EAGAIN | nix::errno::Errno::EINTR) => Ok(false),
            Err(error) => Err(io::Error::other(error)),
        }
    }

    /// Bounded single-packet receive. Unexpected passed file descriptors are
    /// closed immediately; EOF and truncation are never treated as success.
    pub fn try_receive_packet(&self, maximum: usize) -> io::Result<Option<ManagedOutputPacket>> {
        // usize storage supplies the alignment required by cmsghdr. Linux
        // SCM_RIGHTS is limited to 253 FDs; this bounded area allows those,
        // SCM_CREDENTIALS, and room for unexpected ancillary headers.
        let mut control = [0_usize; 2048];
        self.try_receive_packet_with_control(maximum, &mut control)
    }

    fn try_receive_packet_with_control(
        &self,
        maximum: usize,
        control: &mut [usize],
    ) -> io::Result<Option<ManagedOutputPacket>> {
        require_no_implicit_pidfd(self.channel.as_raw_fd())?;
        if maximum == 0 || maximum > 4096 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid managed output packet bound",
            ));
        }
        let mut data = vec![0_u8; maximum + 1];
        let mut iov = libc::iovec {
            iov_base: data.as_mut_ptr().cast(),
            iov_len: data.len(),
        };
        // SAFETY: an all-zero msghdr is a valid empty C message descriptor.
        let mut message: libc::msghdr = unsafe { std::mem::zeroed() };
        message.msg_iov = &mut iov;
        message.msg_iovlen = 1;
        message.msg_control = control.as_mut_ptr().cast();
        message.msg_controllen = std::mem::size_of_val(control);
        // SAFETY: the message points only to live, writable buffers whose
        // lengths are set above; the kernel validates the socket descriptor.
        let count = unsafe {
            libc::recvmsg(
                self.channel.as_raw_fd(),
                &mut message,
                libc::MSG_DONTWAIT | libc::MSG_CMSG_CLOEXEC,
            )
        };
        if count < 0 {
            let error = io::Error::last_os_error();
            return if matches!(
                error.kind(),
                io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
            ) {
                Ok(None)
            } else {
                Err(error)
            };
        }
        let count = count as usize;
        let invalid_packet = message.msg_flags & (libc::MSG_TRUNC | libc::MSG_CTRUNC) != 0
            || count == 0
            || count > maximum;
        let available = message.msg_controllen.min(std::mem::size_of_val(control));
        let (credentials, invalid_control) = parse_control_and_close_rights(control, available);
        if invalid_packet || invalid_control {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid or truncated managed output packet",
            ));
        }
        let credentials = credentials.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::PermissionDenied,
                "missing managed output packet credentials",
            )
        })?;
        data.truncate(count);
        Ok(Some(ManagedOutputPacket {
            bytes: data,
            credentials,
        }))
    }
}

impl Drop for ManagedOutputInheritedFds {
    fn drop(&mut self) {
        // Shutdown affects the socket itself, including inherited dup FDs.
        // A later wrapper cannot send a request and consume an old ACK.
        match shutdown(self.channel.as_raw_fd(), Shutdown::Both) {
            Ok(()) => {}
            Err(_) => {
                // An unproven shutdown must not allow this cookie to be
                // reissued after a different dup closes or reuses its number.
                self.shutdown_unknown.store(true, Ordering::Release);
            }
        }
    }
}

#[cfg(test)]
mod kernel_tests;

/// Always inspect the actual kernel-written ancillary prefix, including when
/// MSG_CTRUNC is set: any SCM_RIGHTS descriptors that fit were installed and
/// must be closed even though the enclosing packet will be rejected.
fn parse_control_and_close_rights(
    control: &[usize],
    available: usize,
) -> (Option<ManagedOutputPeerCredentials>, bool) {
    let header_size = std::mem::size_of::<libc::cmsghdr>();
    let alignment = std::mem::size_of::<usize>();
    let aligned_header = (header_size + alignment - 1) & !(alignment - 1);
    let bytes = control.as_ptr().cast::<u8>();
    let mut offset = 0_usize;
    let mut credentials = None;
    let mut invalid = false;
    while offset
        .checked_add(header_size)
        .is_some_and(|end| end <= available)
    {
        // SAFETY: the checked offset and header length are inside the
        // kernel-written prefix; read_unaligned imposes no alignment premise.
        let header = unsafe { std::ptr::read_unaligned(bytes.add(offset).cast::<libc::cmsghdr>()) };
        let length = header.cmsg_len as usize;
        if length < aligned_header {
            invalid = true;
            break;
        }
        let actual = length.min(available - offset);
        if actual < aligned_header {
            invalid = true;
            break;
        }
        let body = actual - aligned_header;
        // SAFETY: aligned_header was checked against the available prefix.
        let contents = unsafe { bytes.add(offset + aligned_header) };
        if header.cmsg_level == libc::SOL_SOCKET && header.cmsg_type == libc::SCM_RIGHTS {
            invalid = true;
            let size = std::mem::size_of::<RawFd>();
            for i in 0..(body / size) {
                // SAFETY: each complete integer lies in the checked prefix.
                let fd =
                    unsafe { std::ptr::read_unaligned(contents.add(i * size).cast::<RawFd>()) };
                // SAFETY: the kernel installed each received SCM_RIGHTS fd in
                // this process; it is not transferred to any other owner.
                unsafe {
                    libc::close(fd);
                }
            }
            if body % size != 0 {
                invalid = true;
            }
        } else if header.cmsg_level == libc::SOL_SOCKET
            && header.cmsg_type == libc::SCM_CREDENTIALS
            && body == std::mem::size_of::<libc::ucred>()
            && credentials.is_none()
        {
            // SAFETY: exact ucred size was checked inside the received body.
            let value = unsafe { std::ptr::read_unaligned(contents.cast::<libc::ucred>()) };
            credentials = Some(ManagedOutputPeerCredentials {
                pid: value.pid,
                uid: value.uid,
                gid: value.gid,
            });
        } else {
            invalid = true;
        }
        if length > available - offset {
            invalid = true;
            break;
        }
        let Some(step) = length
            .checked_add(alignment - 1)
            .map(|v| v & !(alignment - 1))
        else {
            invalid = true;
            break;
        };
        let Some(next) = offset.checked_add(step) else {
            invalid = true;
            break;
        };
        if next <= offset {
            invalid = true;
            break;
        }
        offset = next;
    }
    (credentials, invalid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nix::sys::socket::{socketpair, AddressFamily, SockFlag};
    use std::os::unix::net::UnixStream;

    #[test]
    fn inherited_seqpacket_retains_original_directory_and_kernel_sender() {
        let directory = File::open(".").expect("existing test working directory");
        let (child, parent) = socketpair(
            AddressFamily::Unix,
            SockType::SeqPacket,
            None,
            SockFlag::SOCK_CLOEXEC,
        )
        .expect("seqpacket pair");
        let owner =
            ManagedOutputInheritedFds::seal_exclusive(directory.as_raw_fd(), child.as_raw_fd())
                .expect("exclusive descriptors");
        assert_eq!(
            ManagedOutputInheritedFds::seal_exclusive(directory.as_raw_fd(), child.as_raw_fd())
                .err()
                .map(|error| error.kind()),
            Some(io::ErrorKind::AlreadyExists),
            "one original socket cannot have two lease control owners"
        );
        let metadata = directory.metadata().expect("original directory stat");
        assert_eq!(
            owner.directory_identity().expect("retained stat"),
            (metadata.dev(), metadata.ino())
        );
        send(parent.as_raw_fd(), b"response", MsgFlags::empty()).expect("single packet");
        let packet = owner
            .try_receive_packet(64)
            .expect("packet observation")
            .expect("packet ready");
        assert_eq!(packet.bytes, b"response");
        assert_eq!(packet.credentials.pid, nix::unistd::getpid().as_raw());
        assert_eq!(packet.credentials.uid, nix::unistd::getuid().as_raw());
        drop(parent);
        assert!(
            owner.try_receive_packet(64).is_err(),
            "EOF is not an authenticated reply"
        );
        drop(owner);
        assert_eq!(
            ManagedOutputInheritedFds::seal_exclusive(directory.as_raw_fd(), child.as_raw_fd())
                .err()
                .map(|error| error.kind()),
            Some(io::ErrorKind::AlreadyExists),
            "a dropped wrapper cannot reissue the still-open original socket"
        );
        assert!(
            directory.metadata().is_ok(),
            "sealing never takes caller's original descriptor"
        );
    }

    #[test]
    fn non_seqpacket_is_rejected_without_consuming_the_original() {
        let directory = File::open(".").expect("existing test working directory");
        let (stream, _other) = UnixStream::pair().expect("stream pair");
        assert!(ManagedOutputInheritedFds::seal_exclusive(
            directory.as_raw_fd(),
            stream.as_raw_fd()
        )
        .is_err());
        assert!(directory.metadata().is_ok());
        assert!(stream.peer_addr().is_ok());
    }

    #[test]
    fn inherited_socket_with_implicit_pidfd_is_rejected_when_supported() {
        let directory = File::open(".").expect("existing working directory");
        let (child, _parent) = socketpair(
            AddressFamily::Unix,
            SockType::SeqPacket,
            None,
            SockFlag::SOCK_CLOEXEC,
        )
        .expect("seqpacket pair");
        let enabled = 1_i32;
        // SAFETY: the integer pointer is valid for the declared i32 size and
        // the socket remains owned by the test.
        let result = unsafe {
            libc::setsockopt(
                child.as_raw_fd(),
                libc::SOL_SOCKET,
                libc::SO_PASSPIDFD,
                (&enabled as *const i32).cast(),
                std::mem::size_of::<i32>() as libc::socklen_t,
            )
        };
        if result != 0 {
            assert_eq!(
                io::Error::last_os_error().raw_os_error(),
                Some(libc::ENOPROTOOPT)
            );
            return;
        }
        assert_eq!(
            ManagedOutputInheritedFds::seal_exclusive(directory.as_raw_fd(), child.as_raw_fd())
                .err()
                .map(|error| error.kind()),
            Some(io::ErrorKind::InvalidInput)
        );
        assert!(directory.metadata().is_ok());
    }

    #[test]
    fn truncated_ancillary_prefix_closes_the_installed_rights() {
        let directory = File::open(".").expect("existing working directory");
        // SAFETY: dup creates a fresh descriptor owned by this test.
        let received_fd = unsafe { libc::dup(directory.as_raw_fd()) };
        assert!(received_fd >= 0);
        let mut control = [0_usize; 8];
        let alignment = std::mem::size_of::<usize>();
        let header_size = std::mem::size_of::<libc::cmsghdr>();
        let aligned_header = (header_size + alignment - 1) & !(alignment - 1);
        let available = aligned_header + std::mem::size_of::<RawFd>();
        let header = libc::cmsghdr {
            // A larger declared message models the kernel-written prefix when
            // the ancillary buffer was truncated after an installed FD.
            cmsg_len: available + std::mem::size_of::<RawFd>(),
            cmsg_level: libc::SOL_SOCKET,
            cmsg_type: libc::SCM_RIGHTS,
        };
        // SAFETY: the checked fixed storage contains the whole header and FD,
        // and the parser is the sole owner of the duplicated descriptor.
        unsafe {
            std::ptr::write_unaligned(control.as_mut_ptr().cast::<libc::cmsghdr>(), header);
            std::ptr::write_unaligned(
                control
                    .as_mut_ptr()
                    .cast::<u8>()
                    .add(aligned_header)
                    .cast::<RawFd>(),
                received_fd,
            );
        }
        let (_, invalid) = parse_control_and_close_rights(&control, available);
        assert!(invalid);
        // SAFETY: querying an invalid descriptor has no ownership effect.
        assert_eq!(unsafe { libc::fcntl(received_fd, libc::F_GETFD) }, -1);
        assert_eq!(io::Error::last_os_error().raw_os_error(), Some(libc::EBADF));
        assert!(directory.metadata().is_ok());
    }
}
