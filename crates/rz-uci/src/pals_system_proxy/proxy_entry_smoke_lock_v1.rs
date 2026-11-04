//! Bounded proxy-local initial lock admission for the NoGame smoke path.
//! This retains the proxy's original root and lock read FDs. It does not prove
//! the role listener, arena's file-description identity, child executable,
//! source, assets, provider readiness, native admission or GPU completion.

use crate::pals_adaptive_startup::{
    read_control_owned, relative_control_path, verify_root_identity,
};
use cap_fs_ext::{DirExt, MetadataExt as CapMetadataExt};
use cap_std::fs::Dir;
use rz_experiments::PalsSystemStartupLockV6;
use std::{
    fs::{File, Metadata},
    io,
    os::unix::fs::{FileExt, MetadataExt as StdMetadataExt},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
fn alive(until: Instant, cancelled: &AtomicBool) -> io::Result<()> {
    if cancelled.load(Ordering::Acquire) {
        return Err(io::Error::new(
            io::ErrorKind::Interrupted,
            "proxy lock admission cancelled",
        ));
    }
    if Instant::now() >= until {
        return Err(io::Error::new(
            io::ErrorKind::TimedOut,
            "proxy local startup cap expired",
        ));
    }
    Ok(())
}
fn unchanged(a: &Metadata, b: &Metadata) -> bool {
    (
        StdMetadataExt::dev(a),
        StdMetadataExt::ino(a),
        a.len(),
        StdMetadataExt::mtime(a),
        StdMetadataExt::mtime_nsec(a),
        StdMetadataExt::ctime(a),
        StdMetadataExt::ctime_nsec(a),
    ) == (
        StdMetadataExt::dev(b),
        StdMetadataExt::ino(b),
        b.len(),
        StdMetadataExt::mtime(b),
        StdMetadataExt::mtime_nsec(b),
        StdMetadataExt::ctime(b),
        StdMetadataExt::ctime_nsec(b),
    )
}

/// Local descriptor owner acquired only after the existing bounded no-follow
/// reader returns. Arena's original snapshot FD remains a separate owner.
pub(super) struct OriginalSmokeLockPin {
    root: Dir,
    file: File,
    bytes: Vec<u8>,
    metadata: Metadata,
    private_cwd: PathBuf,
    relative: String,
}

pub(super) struct OriginalSmokeLockAdmission {
    pin: OriginalSmokeLockPin,
    lock: PalsSystemStartupLockV6,
    entry: Instant,
    local_until: Instant,
    cancelled: Arc<AtomicBool>,
}

/// If the bounded reader never returned a file, only its opened root may be
/// present. A successfully acquired pin is returned intact on later failures.
pub(super) struct SmokeLockAdmissionFailure {
    pub error: io::Error,
    pub root: Option<Dir>,
    pub pin: Option<OriginalSmokeLockPin>,
}

impl OriginalSmokeLockAdmission {
    /// This early read uses only caller-local entry/startup cap, not a claim
    /// that either original arena grant has been obtained. Subsequent session
    /// work must intersect with the authenticated original run/cleanup grants.
    pub(super) fn open(
        lock_path: &Path,
        expected_sha256: &str,
        startup_ms: u64,
        private_cwd: &Path,
        entry: Instant,
        cancelled: &Arc<AtomicBool>,
    ) -> Result<Self, SmokeLockAdmissionFailure> {
        let no_owner = |error| SmokeLockAdmissionFailure {
            error,
            root: None,
            pin: None,
        };
        if entry > Instant::now() || startup_ms == 0 || !private_cwd.is_absolute() {
            return Err(no_owner(invalid(
                "proxy lock local admission bounds invalid",
            )));
        }
        let Some(local_until) = entry.checked_add(Duration::from_millis(startup_ms)) else {
            return Err(no_owner(invalid("proxy lock local startup cap overflow")));
        };
        if let Err(error) = alive(local_until, cancelled) {
            return Err(no_owner(error));
        }
        let relative = match relative_control_path(lock_path, private_cwd) {
            Ok(relative) => relative,
            Err(error) => return Err(no_owner(io::Error::other(error))),
        };
        let root = match Dir::open_ambient_dir(private_cwd, cap_std::ambient_authority()) {
            Ok(root) => root,
            Err(error) => return Err(no_owner(error)),
        };
        if let Err(error) = verify_root_identity(&root, private_cwd) {
            return Err(SmokeLockAdmissionFailure {
                error: io::Error::other(error),
                root: Some(root),
                pin: None,
            });
        }
        if let Err(error) = alive(local_until, cancelled) {
            return Err(SmokeLockAdmissionFailure {
                error,
                root: Some(root),
                pin: None,
            });
        }
        let (file, bytes, metadata) = match read_control_owned(&root, &relative) {
            Ok(value) => value,
            Err(error) => {
                return Err(SmokeLockAdmissionFailure {
                    error: io::Error::other(error),
                    root: Some(root),
                    pin: None,
                });
            }
        };
        let pin = OriginalSmokeLockPin {
            root,
            file,
            bytes,
            metadata,
            private_cwd: private_cwd.to_owned(),
            relative,
        };
        let checked = (|| {
            alive(local_until, cancelled)?;
            pin.verify_held(local_until, cancelled)?;
            let text = std::str::from_utf8(&pin.bytes).map_err(io::Error::other)?;
            let lock = PalsSystemStartupLockV6::from_json(text).map_err(io::Error::other)?;
            if lock.canonical_sha256 != expected_sha256
                || lock.config.native_options.startup_ms != startup_ms
            {
                return Err(invalid("proxy lock differs from its local declaration"));
            }
            alive(local_until, cancelled)?;
            Ok(lock)
        })();
        match checked {
            Ok(lock) => Ok(Self {
                pin,
                lock,
                entry,
                local_until,
                cancelled: Arc::clone(cancelled),
            }),
            Err(error) => Err(SmokeLockAdmissionFailure {
                error,
                root: None,
                pin: Some(pin),
            }),
        }
    }

    pub(super) fn lock(&self) -> &PalsSystemStartupLockV6 {
        &self.lock
    }
    pub(super) fn local_until(&self) -> Instant {
        self.local_until
    }
    pub(super) fn entry(&self) -> Instant {
        self.entry
    }
    pub(super) fn matches_token(&self, cancelled: &Arc<AtomicBool>) -> bool {
        Arc::ptr_eq(&self.cancelled, cancelled)
    }
    pub(super) fn verify_held(&self) -> io::Result<()> {
        alive(self.local_until, &self.cancelled)?;
        self.pin.verify_held(self.local_until, &self.cancelled)?;
        alive(self.local_until, &self.cancelled)
    }
}

impl OriginalSmokeLockPin {
    /// Read only through the exact previously returned descriptor. The root
    /// and FD are retained even if any recheck fails; no replacement is opened.
    fn verify_held(&self, until: Instant, cancelled: &AtomicBool) -> io::Result<()> {
        alive(until, cancelled)?;
        verify_root_identity(&self.root, &self.private_cwd).map_err(io::Error::other)?;
        self.verify_name()?;
        let before = self.file.metadata()?;
        if !before.is_file()
            || !unchanged(&self.metadata, &before)
            || before.len() != self.bytes.len() as u64
        {
            return Err(invalid("proxy original lock identity changed"));
        }
        let mut offset = 0usize;
        let mut buffer = [0; 8192];
        while offset < self.bytes.len() {
            alive(until, cancelled)?;
            let limit = buffer.len().min(self.bytes.len() - offset);
            let read = match self.file.read_at(&mut buffer[..limit], offset as u64) {
                Ok(read) => read,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error),
            };
            if read == 0 || buffer[..read] != self.bytes[offset..offset + read] {
                return Err(invalid("proxy original lock bytes changed"));
            }
            offset += read;
        }
        let after = self.file.metadata()?;
        if !unchanged(&before, &after) {
            return Err(invalid("proxy original lock changed during verification"));
        }
        self.verify_name()?;
        verify_root_identity(&self.root, &self.private_cwd).map_err(io::Error::other)?;
        alive(until, cancelled)
    }

    /// Resolve directory components without following links and compare the
    /// named inode to the original FD; never replace that FD with a reopen.
    fn verify_name(&self) -> io::Result<()> {
        let (parents, name) = self
            .relative
            .rsplit_once('/')
            .ok_or_else(|| invalid("proxy lock parent missing"))?;
        let mut directory = self.root.try_clone()?;
        for component in parents.split('/') {
            directory = directory.open_dir_nofollow(component)?;
        }
        let named = directory.symlink_metadata(name)?;
        if !named.is_file()
            || CapMetadataExt::dev(&named) != StdMetadataExt::dev(&self.metadata)
            || CapMetadataExt::ino(&named) != StdMetadataExt::ino(&self.metadata)
            || named.len() != self.metadata.len()
        {
            return Err(invalid("proxy original lock name changed"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn local_cap_and_cancellation_are_independent() {
        let token = AtomicBool::new(false);
        assert!(alive(Instant::now() + Duration::from_secs(60), &token).is_ok());
        token.store(true, Ordering::Release);
        assert_eq!(
            alive(Instant::now() + Duration::from_secs(60), &token)
                .unwrap_err()
                .kind(),
            io::ErrorKind::Interrupted
        );
        token.store(false, Ordering::Release);
        assert_eq!(
            alive(Instant::now(), &token).unwrap_err().kind(),
            io::ErrorKind::TimedOut
        );
    }
}
