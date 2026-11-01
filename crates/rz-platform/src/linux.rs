//! Current-thread observations and the explicit bounded libc affinity boundary.
use crate::placement::{ThreadAffinityAccess, ThreadAffinityMask, ThreadAffinitySnapshot};
use crate::{LinuxThreadAffinity, PlatformCpuError};
use nix::libc;
use std::collections::BTreeSet;
use std::io::Read;
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::net::UnixDatagram;

const MAX_AFFINITY_CPUS: usize = 4096;
const MASK_WORDS: usize = MAX_AFFINITY_CPUS / usize::BITS as usize;

extern "C" {
    fn sched_getaffinity(pid: i32, cpusetsize: usize, mask: *mut usize) -> i32;
    fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const usize) -> i32;
}

/// Duplicate an explicitly supplied inherited FD without taking ownership of
/// the original. The caller must still validate socket type and peer identity.
/// The original remains owned by the process that inherited it; only the new
/// close-on-exec descriptor is owned by the returned value.
pub fn duplicate_inherited_unix_datagram(fd: i32) -> std::io::Result<UnixDatagram> {
    if fd < 3 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "inherited descriptor must not be standard I/O",
        ));
    }
    // The kernel validates fd and duplicates it atomically with close-on-exec;
    // it never transfers or closes the inherited original.
    let duplicated = unsafe { libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, 3_i32) };
    if duplicated < 0 {
        return Err(std::io::Error::last_os_error());
    }
    // Check the kernel socket type before creating a typed UnixDatagram.
    let mut kind = 0_i32;
    let mut domain = 0_i32;
    let mut kind_len = std::mem::size_of::<i32>() as libc::socklen_t;
    let mut domain_len = kind_len;
    // SAFETY: both output pointers are valid writable i32 storage and lengths
    // describe that storage; duplicated is a valid descriptor owned here.
    let type_ok = unsafe {
        libc::getsockopt(
            duplicated,
            libc::SOL_SOCKET,
            libc::SO_TYPE,
            (&mut kind as *mut i32).cast(),
            &mut kind_len,
        )
    } == 0;
    let domain_ok = unsafe {
        libc::getsockopt(
            duplicated,
            libc::SOL_SOCKET,
            libc::SO_DOMAIN,
            (&mut domain as *mut i32).cast(),
            &mut domain_len,
        )
    } == 0;
    if !type_ok
        || !domain_ok
        || kind_len as usize != std::mem::size_of::<i32>()
        || domain_len as usize != std::mem::size_of::<i32>()
        || kind != libc::SOCK_DGRAM
        || domain != libc::AF_UNIX
    {
        // SAFETY: duplicated is still our uniquely owned kernel descriptor.
        unsafe {
            libc::close(duplicated);
        }
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "inherited descriptor is not a Unix datagram socket",
        ));
    }
    // SAFETY: successful F_DUPFD_CLOEXEC returned a new, open Unix datagram
    // descriptor that has not been handed to another Rust owner.
    Ok(unsafe { UnixDatagram::from_raw_fd(duplicated) })
}

/// Mark the explicitly child-exclusive inherited descriptor close-on-exec,
/// after confirming it still names the same object as the verified duplicate.
/// This changes descriptor flags only; it does not close the original or alter
/// the shared file-status flags. The caller must own the inheritance contract
/// and prevent concurrent close/reuse of `fd` for the duration of this call.
pub fn mark_inherited_datagram_cloexec(
    fd: i32,
    verified_duplicate: &UnixDatagram,
) -> std::io::Result<()> {
    if fd < 3 || fd == verified_duplicate.as_raw_fd() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "invalid original descriptor",
        ));
    }
    let mut original = std::mem::MaybeUninit::<libc::stat>::uninit();
    let mut duplicate = std::mem::MaybeUninit::<libc::stat>::uninit();
    // SAFETY: fstat receives valid writable stat storage; on failure neither
    // value is read and descriptor ownership remains unchanged.
    if unsafe { libc::fstat(fd, original.as_mut_ptr()) } != 0
        || unsafe { libc::fstat(verified_duplicate.as_raw_fd(), duplicate.as_mut_ptr()) } != 0
    {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: each fstat above returned success and initialized its value.
    let original = unsafe { original.assume_init() };
    let duplicate = unsafe { duplicate.assume_init() };
    if (original.st_dev, original.st_ino, original.st_mode)
        != (duplicate.st_dev, duplicate.st_ino, duplicate.st_mode)
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "original descriptor changed",
        ));
    }
    // F_GETFD/F_SETFD operate on per-descriptor flags, not F_GETFL/F_SETFL.
    // SAFETY: both calls accept an integer descriptor and integer flags.
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFD) };
    if flags < 0 {
        return Err(std::io::Error::last_os_error());
    }
    if flags & libc::FD_CLOEXEC == 0
        && unsafe { libc::fcntl(fd, libc::F_SETFD, flags | libc::FD_CLOEXEC) } < 0
    {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

fn affinity_native_error(function: &'static str) -> PlatformCpuError {
    match std::io::Error::last_os_error().raw_os_error() {
        Some(os_code) => PlatformCpuError::NativeCall {
            function,
            os_code: os_code as u32,
        },
        None => PlatformCpuError::ReadUnavailable,
    }
}
fn native_current_mask() -> Result<Vec<u16>, PlatformCpuError> {
    let mut words = [0usize; MASK_WORDS];
    // SAFETY: pid=0 addresses only the calling thread. The initialized native
    // unsigned-long array is aligned and writable for exactly cpusetsize bytes.
    // The libc interface returns 0, unlike the raw syscall's byte-count return.
    if unsafe { sched_getaffinity(0, std::mem::size_of_val(&words), words.as_mut_ptr()) } != 0 {
        return Err(affinity_native_error("sched_getaffinity"));
    }
    let mut cpus = Vec::new();
    for cpu in 0..MAX_AFFINITY_CPUS {
        if words[cpu / usize::BITS as usize] & (1usize << (cpu % usize::BITS as usize)) != 0 {
            cpus.push(cpu as u16);
        }
    }
    if cpus.is_empty() {
        return Err(PlatformCpuError::InvalidNativeBuffer);
    }
    Ok(cpus)
}

pub(crate) struct NativeThreadAffinity;
impl ThreadAffinityAccess for NativeThreadAffinity {
    fn accepts(&self, mask: &ThreadAffinityMask) -> bool {
        matches!(mask, ThreadAffinityMask::Linux { .. })
    }
    fn observe(&mut self) -> Result<ThreadAffinitySnapshot, PlatformCpuError> {
        // Reuse the existing bounded procfs identity/readback and require the
        // independent native mask to agree between both stable observations.
        let before = observe_linux_thread_affinity()?;
        let cpus = native_current_mask()?;
        let after = observe_linux_thread_affinity()?;
        if before != after || before.cpu_ids != cpus {
            return Err(PlatformCpuError::ChangedDuringProbe);
        }
        Ok(ThreadAffinitySnapshot::Linux(before))
    }
    fn set(
        &mut self,
        mask: &ThreadAffinityMask,
    ) -> Result<Option<ThreadAffinityMask>, PlatformCpuError> {
        let ThreadAffinityMask::Linux { cpu_ids } = mask else {
            return Err(PlatformCpuError::UnsupportedArchitecture);
        };
        if cpu_ids.is_empty() || cpu_ids.len() > MAX_AFFINITY_CPUS {
            return Err(PlatformCpuError::InvalidNativeBuffer);
        }
        let mut words = [0usize; MASK_WORDS];
        for &cpu in cpu_ids {
            let cpu = usize::from(cpu);
            if cpu >= MAX_AFFINITY_CPUS {
                return Err(PlatformCpuError::NativeBufferBudget);
            }
            words[cpu / usize::BITS as usize] |= 1usize << (cpu % usize::BITS as usize);
        }
        // SAFETY: pid=0 is the calling thread, never the process leader or a
        // foreign tid. The native unsigned-long bitset has the requested size
        // and remains alive for this synchronous libc call. No pointer escapes.
        if unsafe { sched_setaffinity(0, std::mem::size_of_val(&words), words.as_ptr()) } != 0 {
            return Err(affinity_native_error("sched_setaffinity"));
        }
        Ok(None)
    }
}

fn read_status() -> Result<String, PlatformCpuError> {
    let file = std::fs::File::open("/proc/thread-self/status")
        .map_err(|_| PlatformCpuError::ReadUnavailable)?;
    let mut bytes = Vec::new();
    file.take(64 * 1024 + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| PlatformCpuError::ReadUnavailable)?;
    if bytes.len() > 64 * 1024 {
        return Err(PlatformCpuError::NativeBufferBudget);
    }
    String::from_utf8(bytes).map_err(|_| PlatformCpuError::InvalidText)
}

fn parse_status(raw: &str) -> Result<(u64, Vec<u16>), PlatformCpuError> {
    let one = |key: &str| -> Result<&str, PlatformCpuError> {
        let mut rows = raw.lines().filter_map(|line| line.strip_prefix(key));
        let row = rows.next().ok_or(PlatformCpuError::InvalidText)?.trim();
        if rows.next().is_some() {
            return Err(PlatformCpuError::InvalidText);
        }
        Ok(row)
    };
    let thread_id = one("Pid:")?
        .parse::<u64>()
        .map_err(|_| PlatformCpuError::InvalidText)?;
    if thread_id == 0 {
        return Err(PlatformCpuError::InvalidText);
    }
    let mut cpus = BTreeSet::new();
    let number = |s: &str| -> Result<u16, PlatformCpuError> {
        if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
            return Err(PlatformCpuError::InvalidText);
        }
        let n = s
            .parse::<u16>()
            .map_err(|_| PlatformCpuError::InvalidText)?;
        if n >= 4096 {
            return Err(PlatformCpuError::NativeBufferBudget);
        }
        Ok(n)
    };
    for range in one("Cpus_allowed_list:")?.split(',') {
        let (lo, hi) = if let Some((a, b)) = range.split_once('-') {
            (number(a)?, number(b)?)
        } else {
            let n = number(range)?;
            (n, n)
        };
        if lo > hi {
            return Err(PlatformCpuError::InvalidText);
        }
        for cpu in lo..=hi {
            if !cpus.insert(cpu) {
                return Err(PlatformCpuError::InvalidText);
            }
        }
    }
    if cpus.is_empty() {
        return Err(PlatformCpuError::InvalidText);
    }
    Ok((thread_id, cpus.into_iter().collect()))
}

pub fn observe_linux_thread_affinity() -> Result<LinuxThreadAffinity, PlatformCpuError> {
    let before = parse_status(&read_status()?)?;
    let after = parse_status(&read_status()?)?;
    if before != after {
        return Err(PlatformCpuError::ChangedDuringProbe);
    }
    Ok(LinuxThreadAffinity {
        process_id: std::process::id(),
        thread_id: before.0,
        cpu_ids: before.1,
        identity_stable: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::fd::AsRawFd;

    #[test]
    fn inherited_datagram_duplication_never_closes_original() {
        let (original, peer) = UnixDatagram::pair().unwrap();
        let duplicate = duplicate_inherited_unix_datagram(original.as_raw_fd()).unwrap();
        // Simulate the inheritable FD delivered by the parent and verify only
        // per-descriptor flags change, never shared file-status flags.
        let fd = original.as_raw_fd();
        let original_flags = unsafe { libc::fcntl(fd, libc::F_GETFD) };
        assert!(original_flags >= 0);
        assert_eq!(
            unsafe { libc::fcntl(fd, libc::F_SETFD, original_flags & !libc::FD_CLOEXEC) },
            0
        );
        let status_before = unsafe { libc::fcntl(fd, libc::F_GETFL) };
        assert!(status_before >= 0);
        mark_inherited_datagram_cloexec(fd, &duplicate).unwrap();
        assert_ne!(
            unsafe { libc::fcntl(fd, libc::F_GETFD) } & libc::FD_CLOEXEC,
            0
        );
        assert_eq!(unsafe { libc::fcntl(fd, libc::F_GETFL) }, status_before);
        drop(duplicate);
        original.send(b"owner").unwrap();
        let mut received = [0_u8; 5];
        assert_eq!(peer.recv(&mut received).unwrap(), received.len());
        assert_eq!(&received, b"owner");
        assert!(duplicate_inherited_unix_datagram(-1).is_err());
        let regular = std::fs::File::open("/dev/null").unwrap();
        assert!(duplicate_inherited_unix_datagram(regular.as_raw_fd()).is_err());
    }

    #[test]
    fn current_thread_parser_preserves_large_cpu_ids_and_rejects_ambiguity() {
        assert_eq!(
            parse_status("Pid:\t3\nCpus_allowed_list:\t1,64-65").unwrap(),
            (3, vec![1, 64, 65])
        );
        for raw in [
            "Pid:3\nPid:4\nCpus_allowed_list:0",
            "Pid:0\nCpus_allowed_list:0",
            "Pid:3\nCpus_allowed_list:0,0",
            "Pid:3\nCpus_allowed_list:4096",
        ] {
            assert!(parse_status(raw).is_err());
        }
    }
}
