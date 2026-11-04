//! Linux CPU-only transport check for a socket below a long private root.
//! No model, GPU, production supervisor or inherited child FD is used.

use super::{retire_socket_if_unchanged, socket_transport_path};
use nix::sys::socket::{getsockopt, sockopt::PeerCredentials};
use std::{
    fs::{self, File},
    io::{Read, Write},
    os::{
        fd::AsRawFd,
        unix::{
            ffi::OsStrExt,
            fs::{DirBuilderExt, FileTypeExt, MetadataExt},
            net::{UnixListener, UnixStream},
        },
    },
    path::{Path, PathBuf},
    process::{Child, Command},
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

const CHILD_ROOT: &str = "RZ_TEST_LONG_SOCKET_ROOT";
const PARENT_FD: &str = "RZ_TEST_LONG_SOCKET_PARENT_FD";
const CHILD_TEST: &str = "pals_system_proxy::long_socket_tests::long_socket_child";
static NEXT: AtomicU64 = AtomicU64::new(0);

fn same_directory(first: &std::fs::Metadata, second: &std::fs::Metadata) -> bool {
    first.is_dir() && second.is_dir() && (first.dev(), first.ino()) == (second.dev(), second.ino())
}

/// The child opens the directory in its own process. A leftover matching FD
/// before the open would invalidate the test's no-inheritance premise.
#[test]
fn long_socket_child() {
    let Some(root) = std::env::var_os(CHILD_ROOT).map(PathBuf::from) else {
        return;
    };
    let parent_fd: i32 = std::env::var(PARENT_FD).unwrap().parse().unwrap();
    let named = fs::metadata(&root).unwrap();
    if let Ok(inherited) = File::open(format!("/proc/self/fd/{parent_fd}")) {
        assert!(
            !same_directory(&inherited.metadata().unwrap(), &named),
            "parent directory FD was inherited"
        );
    }
    let directory = File::open(&root).unwrap();
    assert!(same_directory(&directory.metadata().unwrap(), &named));
    let absolute = root.join("owner.sock");
    let named_socket = fs::symlink_metadata(&absolute).unwrap();
    assert!(named_socket.file_type().is_socket());
    assert_eq!(named_socket.uid(), nix::unistd::geteuid().as_raw());
    let address = socket_transport_path(&absolute, &directory).unwrap();
    let mut stream = UnixStream::connect(&address).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(15)))
        .unwrap();
    stream
        .set_write_timeout(Some(Duration::from_secs(15)))
        .unwrap();
    let peer = getsockopt(&stream, PeerCredentials).unwrap();
    assert_eq!(peer.pid(), nix::unistd::getppid().as_raw());
    assert_eq!(peer.uid(), nix::unistd::geteuid().as_raw());
    let alias_socket = fs::symlink_metadata(&address).unwrap();
    assert!(alias_socket.file_type().is_socket());
    assert_eq!(alias_socket.uid(), nix::unistd::geteuid().as_raw());
    assert_eq!(
        (alias_socket.dev(), alias_socket.ino()),
        (named_socket.dev(), named_socket.ino())
    );
    let mut byte = [0];
    stream.read_exact(&mut byte).unwrap();
    assert_eq!(&byte, b"P");
    stream.write_all(b"C").unwrap();
}

/// Own only the new fixture root, and terminate only its one child on timeout.
/// If its pathname was replaced, preserve it rather than deleting another tree.
struct Fixture {
    root: PathBuf,
    base: PathBuf,
    identity: (u64, u64),
    child: Option<Child>,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        if let Some(child) = self.child.as_mut() {
            let settled = match child.try_wait() {
                Ok(Some(_)) => true,
                Ok(None) => {
                    let _ = child.kill();
                    let until = Instant::now() + Duration::from_secs(2);
                    loop {
                        match child.try_wait() {
                            Ok(Some(_)) => break true,
                            Ok(None) if Instant::now() < until => {
                                std::thread::sleep(Duration::from_millis(5));
                            }
                            _ => break false,
                        }
                    }
                }
                Err(_) => false,
            };
            if !settled {
                // Preserve the fixture if the exact child did not terminate.
                return;
            }
        }
        if self.root.parent() != Some(self.base.as_path()) {
            return;
        }
        let Ok(metadata) = fs::symlink_metadata(&self.root) else {
            return;
        };
        if metadata.is_dir()
            && !metadata.file_type().is_symlink()
            && (metadata.dev(), metadata.ino()) == self.identity
        {
            let _ = fs::remove_dir_all(&self.root);
        }
    }
}

#[test]
fn long_private_root_uses_separately_opened_directory_fds() {
    let base = std::env::temp_dir().canonicalize().unwrap();
    let root = base.join(format!(
        "rz-v6-long-socket-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::DirBuilder::new().mode(0o700).create(&root).unwrap();
    let metadata = fs::symlink_metadata(&root).unwrap();
    assert_eq!(metadata.uid(), nix::unistd::geteuid().as_raw());
    assert_eq!(metadata.mode() & 0o077, 0, "fixture root must be private");
    let mut fixture = Fixture {
        root: root.clone(),
        base,
        identity: (metadata.dev(), metadata.ino()),
        child: None,
    };
    let mut deep = root.join("owned-segment-00-long");
    fs::create_dir(&deep).unwrap();
    for index in 1..16 {
        if deep.join("owner.sock").as_os_str().as_bytes().len() > 108 {
            break;
        }
        deep.push(format!("owned-segment-{index:02}-long"));
        fs::create_dir(&deep).unwrap();
    }
    let absolute = deep.join("owner.sock");
    assert!(absolute.is_absolute());
    assert!(absolute.as_os_str().as_bytes().len() > 108);
    let directory = File::open(&deep).unwrap();
    let descriptor_flags = nix::fcntl::fcntl(&directory, nix::fcntl::FcntlArg::F_GETFD).unwrap();
    assert!(
        nix::fcntl::FdFlag::from_bits_truncate(descriptor_flags)
            .contains(nix::fcntl::FdFlag::FD_CLOEXEC)
    );
    let different_directory = File::open(&fixture.root).unwrap();
    assert!(socket_transport_path(&absolute, &different_directory).is_err());
    let address = socket_transport_path(&absolute, &directory).unwrap();
    assert!(address.starts_with(Path::new("/proc/self/fd")));
    let listener = UnixListener::bind(&address).unwrap();
    listener.set_nonblocking(true).unwrap();
    let initial_socket = fs::symlink_metadata(&absolute).unwrap();
    assert!(initial_socket.file_type().is_socket());
    assert_eq!(initial_socket.uid(), nix::unistd::geteuid().as_raw());
    let alias_socket = fs::symlink_metadata(&address).unwrap();
    assert!(alias_socket.file_type().is_socket());
    assert_eq!(alias_socket.uid(), nix::unistd::geteuid().as_raw());
    assert_eq!(
        (alias_socket.dev(), alias_socket.ino()),
        (initial_socket.dev(), initial_socket.ino())
    );
    fixture.child = Some(
        Command::new(std::env::current_exe().unwrap())
            .arg("--exact")
            .arg(CHILD_TEST)
            .arg("--nocapture")
            .env(CHILD_ROOT, &deep)
            .env(PARENT_FD, directory.as_raw_fd().to_string())
            .spawn()
            .unwrap(),
    );
    let until = Instant::now() + Duration::from_secs(20);
    let mut stream = loop {
        match listener.accept() {
            Ok((stream, _)) => break stream,
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                assert!(
                    Instant::now() < until,
                    "owned child did not connect before deadline"
                );
                assert!(
                    fixture
                        .child
                        .as_mut()
                        .unwrap()
                        .try_wait()
                        .unwrap()
                        .is_none(),
                    "owned child exited before connecting"
                );
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(error) => panic!("original listener failed: {error}"),
        }
    };
    let peer = getsockopt(&stream, PeerCredentials).unwrap();
    assert_eq!(peer.pid() as u32, fixture.child.as_ref().unwrap().id());
    assert_eq!(peer.uid(), nix::unistd::geteuid().as_raw());
    stream
        .set_read_timeout(Some(Duration::from_secs(15)))
        .unwrap();
    stream
        .set_write_timeout(Some(Duration::from_secs(15)))
        .unwrap();
    stream.write_all(b"P").unwrap();
    let mut byte = [0];
    stream.read_exact(&mut byte).unwrap();
    assert_eq!(&byte, b"C");
    drop(stream);
    let status = loop {
        if let Some(status) = fixture.child.as_mut().unwrap().try_wait().unwrap() {
            break status;
        }
        assert!(
            Instant::now() < until,
            "owned child did not exit before deadline"
        );
        std::thread::sleep(Duration::from_millis(5));
    };
    assert!(status.success());
    drop(listener);
    let final_socket = fs::symlink_metadata(&absolute).unwrap();
    assert!(final_socket.file_type().is_socket());
    assert_eq!(final_socket.uid(), nix::unistd::geteuid().as_raw());
    assert_eq!(
        (final_socket.dev(), final_socket.ino()),
        (initial_socket.dev(), initial_socket.ino())
    );
    retire_socket_if_unchanged(&absolute, initial_socket.dev(), initial_socket.ino()).unwrap();
}

#[test]
fn socket_retirement_does_not_delete_replacement_inode() {
    let base = std::env::temp_dir().canonicalize().unwrap();
    let root = base.join(format!(
        "rz-v6-socket-replacement-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::DirBuilder::new().mode(0o700).create(&root).unwrap();
    let metadata = fs::symlink_metadata(&root).unwrap();
    assert_eq!(metadata.uid(), nix::unistd::geteuid().as_raw());
    assert_eq!(metadata.mode() & 0o077, 0);
    let _fixture = Fixture {
        root: root.clone(),
        base,
        identity: (metadata.dev(), metadata.ino()),
        child: None,
    };
    let directory = File::open(&root).unwrap();
    let selected = root.join("selected.sock");
    let saved = root.join("original.sock");
    let selected_address = socket_transport_path(&selected, &directory).unwrap();
    let original = UnixListener::bind(&selected_address).unwrap();
    let original_metadata = fs::symlink_metadata(&selected).unwrap();
    assert!(original_metadata.file_type().is_socket());
    assert_eq!(original_metadata.uid(), nix::unistd::geteuid().as_raw());
    fs::rename(&selected, &saved).unwrap();
    let moved_metadata = fs::symlink_metadata(&saved).unwrap();
    assert_eq!(
        (moved_metadata.dev(), moved_metadata.ino()),
        (original_metadata.dev(), original_metadata.ino())
    );
    let replacement = UnixListener::bind(&selected_address).unwrap();
    let replacement_metadata = fs::symlink_metadata(&selected).unwrap();
    let replacement_alias = fs::symlink_metadata(&selected_address).unwrap();
    assert!(replacement_metadata.file_type().is_socket());
    assert_eq!(replacement_metadata.uid(), nix::unistd::geteuid().as_raw());
    assert_eq!(
        (replacement_alias.dev(), replacement_alias.ino()),
        (replacement_metadata.dev(), replacement_metadata.ino())
    );
    assert_ne!(
        (replacement_metadata.dev(), replacement_metadata.ino()),
        (original_metadata.dev(), original_metadata.ino())
    );
    assert!(
        retire_socket_if_unchanged(&selected, original_metadata.dev(), original_metadata.ino())
            .is_err()
    );
    let retained = fs::symlink_metadata(&selected).unwrap();
    assert_eq!(
        (retained.dev(), retained.ino()),
        (replacement_metadata.dev(), replacement_metadata.ino())
    );
    let retained_original = fs::symlink_metadata(&saved).unwrap();
    assert_eq!(
        (retained_original.dev(), retained_original.ino()),
        (original_metadata.dev(), original_metadata.ino())
    );
    drop(replacement);
    drop(original);
    // Both files are owned by this fixture; remove each only with its own inode.
    retire_socket_if_unchanged(
        &selected,
        replacement_metadata.dev(),
        replacement_metadata.ino(),
    )
    .unwrap();
    retire_socket_if_unchanged(&saved, original_metadata.dev(), original_metadata.ino()).unwrap();
}
