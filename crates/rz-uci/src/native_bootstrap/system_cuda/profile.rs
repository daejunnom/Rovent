//! Descriptor-owned LC0 CUDA placement-profile directory.
//!
//! A CUDA profiling prefix must stay in the original private output tree.
//! This module creates one exclusive process/epoch leaf and pins its directory
//! descriptor; it never follows a replacement symlink or cleans up on Drop.
//! The caller must hold its output lifecycle lease without concurrent same-UID
//! mutation and retain this owner until the original session is released,
//! including while physical completion is unknown.

use super::NativeConfig;
use nix::{
    fcntl::{OFlag, open, openat},
    sys::stat::{Mode, fchmod, mkdirat},
    unistd::geteuid,
};
use rz_eval::{
    error::{BackendError, CauseCode, FailureKind, FailureStage},
    pals_onnx::virtual_review::VirtualReviewProfileDirectory,
};
use std::{
    fs::{self, File},
    os::unix::fs::MetadataExt,
    path::{Path, PathBuf},
};

fn io_failure(detail: &'static str, cause: impl std::fmt::Display) -> BackendError {
    BackendError::new(FailureKind::Io, FailureStage::Asset, detail)
        .with_external_cause(CauseCode::RuntimePath, &cause)
}

fn invalid(detail: &'static str) -> BackendError {
    BackendError::new(FailureKind::IdentityMismatch, FailureStage::Asset, detail)
}

/// When creation succeeded but later setup failed, retain the original root
/// and any opened leaf. The name alone does not authorize deleting it.
pub(super) struct IncompleteLc0ProfileDirectory {
    pub(super) root: File,
    pub(super) root_path: PathBuf,
    pub(super) root_device: u64,
    pub(super) root_inode: u64,
    pub(super) leaf_name: String,
    pub(super) leaf: Option<File>,
}

pub(super) enum Lc0ProfileCreateError {
    BeforeCreate(BackendError),
    AfterCreate {
        cause: BackendError,
        retained: IncompleteLc0ProfileDirectory,
    },
}

impl std::fmt::Debug for Lc0ProfileCreateError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BeforeCreate(cause) => {
                formatter.debug_tuple("BeforeCreate").field(cause).finish()
            }
            Self::AfterCreate { cause, .. } => formatter
                .debug_struct("AfterCreate")
                .field("cause", cause)
                .field("original_root_retained", &true)
                .finish_non_exhaustive(),
        }
    }
}

/// The source root FD, opened leaf FD and procfd-backed profile owner are kept
/// together. Directory cleanup needs a separate, explicit owner-authorized
/// lifecycle operation; holding this object is not such authorization.
#[must_use]
pub(super) struct Lc0ProfileDirectory {
    root: File,
    root_path: PathBuf,
    root_device: u64,
    root_inode: u64,
    leaf: File,
    leaf_name: String,
    profile: VirtualReviewProfileDirectory,
    prefix: PathBuf,
    device: u64,
    inode: u64,
}

impl Lc0ProfileDirectory {
    pub(super) fn create(
        config: &NativeConfig,
        process_epoch: u64,
    ) -> Result<Self, Lc0ProfileCreateError> {
        if process_epoch == 0 {
            return Err(Lc0ProfileCreateError::BeforeCreate(invalid(
                "LC0 profile process epoch is zero",
            )));
        }
        let root_path = super::super::private_output_root(&config.output_root)
            .map_err(Lc0ProfileCreateError::BeforeCreate)?;
        Self::create_at_root(&root_path, process_epoch)
    }

    fn create_at_root(root_path: &Path, process_epoch: u64) -> Result<Self, Lc0ProfileCreateError> {
        if process_epoch == 0 {
            return Err(Lc0ProfileCreateError::BeforeCreate(invalid(
                "LC0 profile process epoch is zero",
            )));
        }
        let root = File::from(
            open(
                root_path,
                OFlag::O_RDONLY | OFlag::O_DIRECTORY | OFlag::O_NOFOLLOW | OFlag::O_CLOEXEC,
                Mode::empty(),
            )
            .map_err(|error| {
                Lc0ProfileCreateError::BeforeCreate(io_failure(
                    "cannot open LC0 private output root",
                    error,
                ))
            })?,
        );
        let actual = root.metadata().map_err(|error| {
            Lc0ProfileCreateError::BeforeCreate(io_failure(
                "cannot inspect LC0 private output root",
                error,
            ))
        })?;
        let named = fs::symlink_metadata(root_path).map_err(|error| {
            Lc0ProfileCreateError::BeforeCreate(io_failure(
                "cannot recheck LC0 private output root",
                error,
            ))
        })?;
        if !actual.is_dir()
            || !named.is_dir()
            || named.file_type().is_symlink()
            || actual.dev() != named.dev()
            || actual.ino() != named.ino()
            || actual.uid() != geteuid().as_raw()
            || actual.mode() & 0o022 != 0
        {
            return Err(Lc0ProfileCreateError::BeforeCreate(invalid(
                "LC0 private output root identity or ownership differs",
            )));
        }
        let leaf_name = format!("lc0-profile-{}-{process_epoch}", std::process::id());
        // mkdirat is exclusive; an existing leaf is never adopted or removed.
        mkdirat(&root, leaf_name.as_str(), Mode::S_IRWXU).map_err(|error| {
            Lc0ProfileCreateError::BeforeCreate(io_failure(
                "cannot create exclusive LC0 profile directory",
                error,
            ))
        })?;
        let mut pending = IncompleteLc0ProfileDirectory {
            root,
            root_path: root_path.to_path_buf(),
            root_device: actual.dev(),
            root_inode: actual.ino(),
            leaf_name,
            leaf: None,
        };
        let opened = openat(
            &pending.root,
            pending.leaf_name.as_str(),
            OFlag::O_RDONLY | OFlag::O_DIRECTORY | OFlag::O_NOFOLLOW | OFlag::O_CLOEXEC,
            Mode::empty(),
        );
        match opened {
            Ok(fd) => pending.leaf = Some(File::from(fd)),
            Err(error) => {
                return Err(Lc0ProfileCreateError::AfterCreate {
                    cause: io_failure("cannot open original LC0 profile directory", error),
                    retained: pending,
                });
            }
        }
        match Self::finish_pending(&pending) {
            Ok((profile, prefix, device, inode)) => Ok(Self {
                root: pending.root,
                root_path: pending.root_path,
                root_device: pending.root_device,
                root_inode: pending.root_inode,
                leaf: pending.leaf.expect("opened leaf retained"),
                leaf_name: pending.leaf_name,
                profile,
                prefix,
                device,
                inode,
            }),
            Err(cause) => Err(Lc0ProfileCreateError::AfterCreate {
                cause,
                retained: pending,
            }),
        }
    }

    fn finish_pending(
        pending: &IncompleteLc0ProfileDirectory,
    ) -> Result<(VirtualReviewProfileDirectory, PathBuf, u64, u64), BackendError> {
        let root = pending.root.metadata().map_err(|error| {
            io_failure("cannot inspect original LC0 private output root", error)
        })?;
        let named_root = fs::symlink_metadata(&pending.root_path)
            .map_err(|error| io_failure("cannot recheck named LC0 private output root", error))?;
        if !root.is_dir()
            || !named_root.is_dir()
            || named_root.file_type().is_symlink()
            || root.uid() != geteuid().as_raw()
            || root.mode() & 0o022 != 0
            || (root.dev(), root.ino()) != (pending.root_device, pending.root_inode)
            || (named_root.dev(), named_root.ino()) != (pending.root_device, pending.root_inode)
        {
            return Err(invalid(
                "LC0 private output root changed during profile setup",
            ));
        }
        let leaf = pending
            .leaf
            .as_ref()
            .ok_or_else(|| invalid("LC0 profile leaf is absent"))?;
        let before = leaf
            .metadata()
            .map_err(|error| io_failure("cannot inspect LC0 profile leaf", error))?;
        if !before.is_dir() || before.uid() != geteuid().as_raw() {
            return Err(invalid("LC0 profile leaf is not an owned directory"));
        }
        // A restrictive umask may remove owner bits; restore exactly 0700 on
        // this freshly opened leaf and verify the same descriptor afterward.
        fchmod(leaf, Mode::S_IRWXU)
            .map_err(|error| io_failure("cannot set LC0 profile leaf mode", error))?;
        let after = leaf
            .metadata()
            .map_err(|error| io_failure("cannot recheck LC0 profile leaf", error))?;
        if !after.is_dir()
            || after.uid() != before.uid()
            || after.dev() != before.dev()
            || after.ino() != before.ino()
            || after.mode() & 0o7777 != 0o700
        {
            return Err(invalid("LC0 profile leaf identity or mode changed"));
        }
        let duplicate = leaf
            .try_clone()
            .map_err(|error| io_failure("cannot retain LC0 profile capability", error))?;
        let profile = VirtualReviewProfileDirectory::from_owned_directory(duplicate)?;
        let prefix = profile.path().join("placement");
        let alias = fs::metadata(profile.path())
            .map_err(|error| io_failure("cannot inspect LC0 profile procfd", error))?;
        if !alias.is_dir() || alias.dev() != after.dev() || alias.ino() != after.ino() {
            return Err(invalid("LC0 profile procfd differs from original leaf"));
        }
        let named = File::from(
            openat(
                &pending.root,
                pending.leaf_name.as_str(),
                OFlag::O_RDONLY | OFlag::O_DIRECTORY | OFlag::O_NOFOLLOW | OFlag::O_CLOEXEC,
                Mode::empty(),
            )
            .map_err(|error| io_failure("cannot re-open LC0 profile leaf", error))?,
        )
        .metadata()
        .map_err(|error| io_failure("cannot inspect named LC0 profile leaf", error))?;
        if !named.is_dir() || named.dev() != after.dev() || named.ino() != after.ino() {
            return Err(invalid(
                "LC0 profile leaf name no longer denotes the original",
            ));
        }
        Ok((profile, prefix, after.dev(), after.ino()))
    }

    /// Recheck that both the original procfd and its name under the pinned
    /// root still denote this leaf. A replaced name is not adopted or removed.
    pub(super) fn verify_original(&self) -> Result<(), BackendError> {
        let held_root = self.root.metadata().map_err(|error| {
            io_failure("cannot inspect retained LC0 private output root", error)
        })?;
        let named_root = fs::symlink_metadata(&self.root_path)
            .map_err(|error| io_failure("cannot inspect named LC0 private output root", error))?;
        if !held_root.is_dir()
            || !named_root.is_dir()
            || named_root.file_type().is_symlink()
            || held_root.uid() != geteuid().as_raw()
            || held_root.mode() & 0o022 != 0
            || (held_root.dev(), held_root.ino()) != (self.root_device, self.root_inode)
            || (named_root.dev(), named_root.ino()) != (self.root_device, self.root_inode)
        {
            return Err(invalid("LC0 private output root was replaced or changed"));
        }
        let held = self
            .leaf
            .metadata()
            .map_err(|error| io_failure("cannot inspect retained LC0 profile leaf", error))?;
        let fd = openat(
            &self.root,
            self.leaf_name.as_str(),
            OFlag::O_RDONLY | OFlag::O_DIRECTORY | OFlag::O_NOFOLLOW | OFlag::O_CLOEXEC,
            Mode::empty(),
        )
        .map_err(|error| io_failure("cannot reopen original LC0 profile leaf", error))?;
        let named = File::from(fd)
            .metadata()
            .map_err(|error| io_failure("cannot inspect named LC0 profile leaf", error))?;
        let alias = fs::metadata(self.profile.path())
            .map_err(|error| io_failure("cannot inspect retained LC0 profile procfd", error))?;
        if !held.is_dir()
            || !named.is_dir()
            || !alias.is_dir()
            || held.uid() != geteuid().as_raw()
            || held.mode() & 0o7777 != 0o700
            || (held.dev(), held.ino()) != (self.device, self.inode)
            || (named.dev(), named.ino()) != (self.device, self.inode)
            || (alias.dev(), alias.ino()) != (self.device, self.inode)
        {
            return Err(invalid("LC0 profile original leaf was replaced or changed"));
        }
        Ok(())
    }

    /// The caller must keep this object alive while ORT uses this procfd path.
    pub(super) fn prefix(&self) -> &Path {
        &self.prefix
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        os::unix::fs::{DirBuilderExt, symlink},
        sync::atomic::{AtomicU64, Ordering},
        time::{SystemTime, UNIX_EPOCH},
    };

    fn new_root() -> PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let nonce = NEXT.fetch_add(1, Ordering::Relaxed);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "rz-lc0-profile-test-{}-{now}-{nonce}",
            std::process::id()
        ));
        fs::DirBuilder::new().mode(0o700).create(&root).unwrap();
        root
    }

    #[test]
    fn procfd_keeps_original_leaf_and_replacement_is_rejected() {
        let root = new_root();
        let owner = Lc0ProfileDirectory::create_at_root(&root, 1).unwrap();
        owner.verify_original().unwrap();
        let before = fs::metadata(owner.prefix().parent().unwrap()).unwrap();
        let leaf = root.join(&owner.leaf_name);
        let moved = root.join("original");
        fs::rename(&leaf, &moved).unwrap();
        fs::DirBuilder::new().mode(0o700).create(&leaf).unwrap();
        assert!(owner.verify_original().is_err());
        let after = fs::metadata(owner.prefix().parent().unwrap()).unwrap();
        assert_eq!((before.dev(), before.ino()), (after.dev(), after.ino()));
        drop(owner);
        fs::remove_dir(leaf).unwrap();
        fs::remove_dir(moved).unwrap();
        fs::remove_dir(root).unwrap();
    }

    #[test]
    fn existing_leaf_and_symlink_root_are_never_adopted() {
        let root = new_root();
        let leaf = root.join(format!("lc0-profile-{}-2", std::process::id()));
        fs::DirBuilder::new().mode(0o700).create(&leaf).unwrap();
        assert!(matches!(
            Lc0ProfileDirectory::create_at_root(&root, 2),
            Err(Lc0ProfileCreateError::BeforeCreate(_))
        ));
        assert!(leaf.is_dir());
        fs::remove_dir(leaf).unwrap();
        let alias = root.with_extension("link");
        symlink(&root, &alias).unwrap();
        assert!(matches!(
            Lc0ProfileDirectory::create_at_root(&alias, 3),
            Err(Lc0ProfileCreateError::BeforeCreate(_))
        ));
        fs::remove_file(alias).unwrap();
        fs::remove_dir(root).unwrap();
    }

    #[test]
    fn replacing_named_root_does_not_redirect_the_retained_procfd() {
        let root = new_root();
        let owner = Lc0ProfileDirectory::create_at_root(&root, 4).unwrap();
        let before = fs::metadata(owner.prefix().parent().unwrap()).unwrap();
        let moved = root.with_extension("original");
        fs::rename(&root, &moved).unwrap();
        fs::DirBuilder::new().mode(0o700).create(&root).unwrap();
        assert!(owner.verify_original().is_err());
        let after = fs::metadata(owner.prefix().parent().unwrap()).unwrap();
        assert_eq!((before.dev(), before.ino()), (after.dev(), after.ino()));
        let leaf = moved.join(&owner.leaf_name);
        drop(owner);
        fs::remove_dir(leaf).unwrap();
        fs::remove_dir(moved).unwrap();
        fs::remove_dir(root).unwrap();
    }
}
