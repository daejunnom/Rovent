//! Real Linux ancillary transport checks, isolated from other test FDs.
//! These do not grant or prove an output lease or device completion.
use super::*;
use nix::sys::socket::{socketpair, AddressFamily, SockFlag};
use std::process::Command;

const CHILD_MARKER: &str = "ROVEZERO_MANAGED_OUTPUT_ANCILLARY_TEST_CHILD";

/// The child runs only this same test, so other tests cannot concurrently
/// allocate or close descriptors while its /proc/self/fd count is compared.
fn isolated(test: impl FnOnce()) {
    if std::env::var_os(CHILD_MARKER).is_some() {
        test();
        return;
    }
    let thread = std::thread::current();
    let name = thread.name().expect("test harness provides the test name");
    let status = Command::new(std::env::current_exe().expect("test executable"))
        .arg("--exact")
        .arg(name)
        .arg("--nocapture")
        .env(CHILD_MARKER, "1")
        .env("RUST_TEST_THREADS", "1")
        .status()
        .expect("isolated test process starts");
    assert!(
        status.success(),
        "isolated ancillary check failed: {status}"
    );
}

fn fd_count() -> usize {
    std::fs::read_dir("/proc/self/fd")
        .expect("isolated Linux process descriptor directory")
        .count()
}

fn align(value: usize) -> usize {
    let alignment = std::mem::size_of::<usize>();
    (value + alignment - 1) & !(alignment - 1)
}

/// Send real SCM_RIGHTS over the kernel socket; all entries may name the same
/// sender-owned source FD because the receiver gets independently installed FDs.
fn send_rights(socket: RawFd, source: RawFd, count: usize) {
    send_rights_with_payload(socket, source, count, b"x");
}

fn send_rights_with_payload(socket: RawFd, source: RawFd, count: usize, payload: &[u8]) {
    let header_len = align(std::mem::size_of::<libc::cmsghdr>());
    let body_len = count * std::mem::size_of::<RawFd>();
    let control_len = header_len + align(body_len);
    let mut control = vec![0_usize; control_len / std::mem::size_of::<usize>()];
    let header = libc::cmsghdr {
        cmsg_len: header_len + body_len,
        cmsg_level: libc::SOL_SOCKET,
        cmsg_type: libc::SCM_RIGHTS,
    };
    // SAFETY: control is aligned storage of control_len bytes; the header and
    // every FD slot lie within it. The socket and source stay owned by caller.
    unsafe {
        let bytes = control.as_mut_ptr().cast::<u8>();
        std::ptr::write_unaligned(bytes.cast::<libc::cmsghdr>(), header);
        for index in 0..count {
            std::ptr::write_unaligned(
                bytes
                    .add(header_len + index * std::mem::size_of::<RawFd>())
                    .cast::<RawFd>(),
                source,
            );
        }
        let mut iov = libc::iovec {
            iov_base: payload.as_ptr().cast_mut().cast(),
            iov_len: payload.len(),
        };
        let mut msg: libc::msghdr = std::mem::zeroed();
        msg.msg_iov = &mut iov;
        msg.msg_iovlen = 1;
        msg.msg_control = control.as_mut_ptr().cast();
        msg.msg_controllen = control_len;
        assert_eq!(
            libc::sendmsg(socket, &msg, libc::MSG_NOSIGNAL | libc::MSG_DONTWAIT),
            payload.len() as isize,
            "sendmsg: {}",
            io::Error::last_os_error()
        );
    }
}

#[test]
fn kernel_truncation_and_full_rights_are_closed_without_leaks() {
    isolated(|| {
        let directory = File::open(".").expect("working directory");
        let (child, parent) = socketpair(
            AddressFamily::Unix,
            SockType::SeqPacket,
            None,
            SockFlag::SOCK_CLOEXEC,
        )
        .expect("seqpacket pair");
        let owner =
            ManagedOutputInheritedFds::seal_exclusive(directory.as_raw_fd(), child.as_raw_fd())
                .expect("same original socket and directory");
        let baseline = fd_count();

        // SO_PASSCRED places a credential cmsg before received rights. This
        // buffer fits the credentials and some, but not all, passed FDs.
        let bytes = align(std::mem::size_of::<libc::cmsghdr>())
            + align(std::mem::size_of::<libc::ucred>())
            + align(std::mem::size_of::<libc::cmsghdr>())
            + align(2 * std::mem::size_of::<RawFd>());
        let words = bytes / std::mem::size_of::<usize>();
        let mut control = vec![0_usize; words];

        // First observe MSG_CTRUNC from the real kernel syscall directly;
        // the exact same production parser closes the installed prefix.
        send_rights(parent.as_raw_fd(), directory.as_raw_fd(), 32);
        let mut data = [0_u8; 2];
        let mut iov = libc::iovec {
            iov_base: data.as_mut_ptr().cast(),
            iov_len: data.len(),
        };
        // SAFETY: all buffers remain live and writable through recvmsg.
        let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
        msg.msg_iov = &mut iov;
        msg.msg_iovlen = 1;
        msg.msg_control = control.as_mut_ptr().cast();
        msg.msg_controllen = std::mem::size_of_val(control.as_slice());
        // SAFETY: child remains a live socket FD and the pointers above are valid.
        assert_eq!(
            unsafe {
                libc::recvmsg(
                    child.as_raw_fd(),
                    &mut msg,
                    libc::MSG_CMSG_CLOEXEC | libc::MSG_DONTWAIT,
                )
            },
            1
        );
        assert_ne!(
            msg.msg_flags & libc::MSG_CTRUNC,
            0,
            "kernel must report ancillary truncation"
        );
        assert!(
            fd_count() > baseline,
            "the received prefix must contain installed FDs"
        );
        let (_, rejected) = parse_control_and_close_rights(&control, msg.msg_controllen);
        assert!(rejected);
        assert_eq!(
            fd_count(),
            baseline,
            "installed truncated-prefix FDs must close"
        );

        // Exercise the real production receive syscall/parser with small and
        // normal ancillary buffers repeatedly, not a synthetic header.
        for _ in 0..8 {
            send_rights(parent.as_raw_fd(), directory.as_raw_fd(), 32);
            let result = owner.try_receive_packet_with_control(64, &mut control);
            assert_eq!(
                result.err().map(|error| error.kind()),
                Some(io::ErrorKind::InvalidData)
            );
            assert_eq!(
                fd_count(),
                baseline,
                "truncated receive leaked a descriptor"
            );

            send_rights(parent.as_raw_fd(), directory.as_raw_fd(), 8);
            let result = owner.try_receive_packet(64);
            assert_eq!(
                result.err().map(|error| error.kind()),
                Some(io::ErrorKind::InvalidData)
            );
            assert_eq!(
                fd_count(),
                baseline,
                "full ancillary receive leaked a descriptor"
            );
        }
        // Payload truncation must also close every installed descriptor;
        // valid ancillary data does not make a truncated packet acceptable.
        let oversized = vec![b'x'; 128];
        send_rights_with_payload(parent.as_raw_fd(), directory.as_raw_fd(), 8, &oversized);
        let result = owner.try_receive_packet(1);
        assert_eq!(
            result.err().map(|error| error.kind()),
            Some(io::ErrorKind::InvalidData)
        );
        assert_eq!(
            fd_count(),
            baseline,
            "payload-truncated receive leaked a descriptor"
        );
        assert!(
            directory.metadata().is_ok(),
            "sender-owned directory remains open"
        );
        drop(parent);
        assert!(
            owner.try_receive_packet(64).is_err(),
            "peer EOF is not a valid packet"
        );
    });
}
