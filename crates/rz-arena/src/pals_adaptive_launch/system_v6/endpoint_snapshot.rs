//! Borrow the original V6 launch owner's exact endpoint snapshot pins.
//!
//! This verifies held lock bytes and named inodes, not the identity of a
//! future peer, a running executable, provider readiness or physical finish.
//! The proxy sidecar remains a declaration and is not a process capability.

use super::smoke_preparation::ProxyEntrySmokeSnapshotOwnerV1;
use super::{LockedPalsSystemArenaPairV6, PalsSystemProxyInputsV1};
use crate::{
    ArenaError, NativeLaunchOwner,
    native_launch::linux::{self, InputPin},
};
use cap_fs_ext::{FollowSymlinks, OpenOptionsFollowExt};
use cap_std::fs::{Dir, OpenOptions, OpenOptionsExt};
use rz_experiments::{ArtifactRef, NativeEngineRole, PalsSystemStartupLockV6};
use std::{
    collections::BTreeMap,
    fs::{File, Metadata},
    os::unix::fs::{FileExt, MetadataExt},
    path::{Path, PathBuf},
};

/// Borrow only capabilities actually held by one original owner. The
/// selected smoke owner is never converted to a paired execution owner.
#[derive(Clone, Copy)]
struct SelectedInputView<'a> {
    paired: Option<(&'a LockedPalsSystemArenaPairV6, &'a PalsSystemProxyInputsV1)>,
    startup_lock: &'a PalsSystemStartupLockV6,
    lock_artifact: &'a ArtifactRef,
    startup_path: &'a str,
    child_artifact: &'a ArtifactRef,
    child_path: &'a str,
    proxy_artifact: &'a ArtifactRef,
    proxy_path: &'a str,
    directory: &'a Dir,
    input_directory: &'a Dir,
    input_subdirectories: &'a BTreeMap<PathBuf, Dir>,
    cwd: &'a File,
    path: &'a Path,
    pins: &'a [InputPin],
}

impl<'a> SelectedInputView<'a> {
    fn from_pair(
        owner: &'a NativeLaunchOwner<LockedPalsSystemArenaPairV6>,
        role: NativeEngineRole,
    ) -> Result<Self, ArenaError> {
        let snapshot = &owner.snapshot;
        let pair = owner.spec();
        let sidecar = pair
            .proxy_inputs()
            .ok_or_else(|| invalid("PALS V6 original owner has no proxy sidecar"))?;
        sidecar.validate_for_pair(pair)?;
        let endpoint = pair.endpoint(role);
        Ok(Self {
            paired: Some((pair, sidecar)),
            startup_lock: &endpoint.startup_lock,
            lock_artifact: &endpoint.startup_lock_artifact,
            startup_path: &endpoint.startup_snapshot_path,
            child_artifact: &endpoint.startup_lock.config.endpoint.binary,
            child_path: &endpoint.binary_snapshot_path,
            proxy_artifact: &sidecar.proxy_binary,
            proxy_path: &sidecar.proxy_snapshot_path,
            directory: &snapshot.directory,
            input_directory: &snapshot.input_directory,
            input_subdirectories: &snapshot.input_subdirectories,
            cwd: &snapshot.cwd,
            path: &snapshot.path,
            pins: &snapshot.pins,
        })
    }

    fn from_smoke(
        owner: &'a ProxyEntrySmokeSnapshotOwnerV1,
        role: NativeEngineRole,
    ) -> Result<Self, ArenaError> {
        if owner.role != role
            || owner.plan.input().declared_role != role
            || !owner.execution.matches_plan(&owner.plan)
        {
            return Err(invalid("PALS smoke selected role differs"));
        }
        let plan = owner.plan.input();
        let endpoint = &plan.endpoint;
        Ok(Self {
            paired: None,
            startup_lock: &endpoint.startup_lock,
            lock_artifact: &endpoint.startup_lock_artifact,
            startup_path: &endpoint.startup_snapshot_path,
            child_artifact: &endpoint.startup_lock.config.endpoint.binary,
            child_path: &endpoint.binary_snapshot_path,
            proxy_artifact: &plan.proxy_binary,
            proxy_path: &plan.proxy_snapshot_path,
            directory: &owner.directory,
            input_directory: &owner.input_directory,
            input_subdirectories: &owner.input_subdirectories,
            cwd: &owner.cwd,
            path: &owner.path,
            pins: &owner.pins,
        })
    }
}

/// Check both endpoint views against the same original owner before the V6
/// supervisor proceeds. This is a pinned-input observation, not a spawn,
/// independent executable attestation, receipt, or physical-completion proof.
pub(crate) fn verify_original_proxy_snapshots(
    owner: &NativeLaunchOwner<LockedPalsSystemArenaPairV6>,
) -> Result<(), ArenaError> {
    for role in [NativeEngineRole::Baseline, NativeEngineRole::Candidate] {
        PalsV6EndpointSnapshot::from_original(owner, role)?;
    }
    Ok(())
}

fn invalid(detail: &'static str) -> ArenaError {
    ArenaError::Integrity(detail.into())
}

/// A borrow from the actual owner, not a copy or an independently reopened
/// replacement. The caller must retain this owner through its physical scope.
pub(crate) struct PalsV6EndpointSnapshot<'a> {
    view: SelectedInputView<'a>,
    role: NativeEngineRole,
    lock: &'a InputPin,
    proxy: &'a InputPin,
    child: &'a InputPin,
}

impl<'a> PalsV6EndpointSnapshot<'a> {
    /// Resolve only pins already held by this original owner. A sidecar whose
    /// proxy was not registered as a snapshot input is an explicit unsupported
    /// V6 execution path, not permission to reopen the source executable.
    pub(crate) fn from_original(
        owner: &'a NativeLaunchOwner<LockedPalsSystemArenaPairV6>,
        role: NativeEngineRole,
    ) -> Result<Self, ArenaError> {
        Self::from_view(SelectedInputView::from_pair(owner, role)?, role)
    }

    pub(crate) fn from_smoke(
        owner: &'a ProxyEntrySmokeSnapshotOwnerV1,
        role: NativeEngineRole,
    ) -> Result<Self, ArenaError> {
        Self::from_view(SelectedInputView::from_smoke(owner, role)?, role)
    }

    fn from_view(view: SelectedInputView<'a>, role: NativeEngineRole) -> Result<Self, ArenaError> {
        let lock = linux::pin(view.pins, view.lock_artifact)?;
        let child = linux::pin(view.pins, view.child_artifact)?;
        let proxy = linux::pin(view.pins, view.proxy_artifact).map_err(|_| {
            ArenaError::Invalid("PALS V6 proxy is not registered in the original snapshot".into())
        })?;
        let value = Self {
            view,
            role,
            lock,
            proxy,
            child,
        };
        value.verify_unchanged()?;
        Ok(value)
    }

    /// Repeat immediately before the original owner consumes these pins.
    /// Separate snapshot copies are not created; executable bytes rely on the
    /// existing full prelaunch verification of this same NativeLaunchOwner.
    pub(crate) fn verify_unchanged(&self) -> Result<(), ArenaError> {
        if let Some((pair, sidecar)) = self.view.paired {
            sidecar.validate_for_pair(pair)?;
        }
        let snapshot = &self.view;
        verify_snapshot_root(snapshot)?;
        linux::verify_input_directories(
            &snapshot.directory,
            &snapshot.input_directory,
            &snapshot.input_subdirectories,
        )?;
        verify_named_pin(snapshot, self.lock, snapshot.startup_path, false)?;
        verify_named_pin(snapshot, self.child, snapshot.child_path, true)?;
        verify_named_pin(snapshot, self.proxy, snapshot.proxy_path, true)?;
        let bytes = snapshot.startup_lock.canonical_bytes()?;
        verify_held_bytes(&self.lock.file, &bytes)?;
        // Recheck names and ancestors after the read; this is an observation,
        // not a claim that an independently writable filesystem is immutable.
        verify_snapshot_root(snapshot)?;
        linux::verify_input_directories(
            &snapshot.directory,
            &snapshot.input_directory,
            &snapshot.input_subdirectories,
        )?;
        verify_named_pin(snapshot, self.lock, snapshot.startup_path, false)?;
        verify_named_pin(snapshot, self.child, snapshot.child_path, true)?;
        verify_named_pin(snapshot, self.proxy, snapshot.proxy_path, true)?;
        Ok(())
    }

    pub(crate) fn lock_file(&self) -> &File {
        &self.lock.file
    }

    pub(crate) fn proxy_file(&self) -> &File {
        &self.proxy.file
    }

    pub(crate) fn child_file(&self) -> &File {
        &self.child.file
    }

    pub(crate) fn role(&self) -> NativeEngineRole {
        self.role
    }
}

fn verify_snapshot_root(snapshot: &SelectedInputView<'_>) -> Result<(), ArenaError> {
    if !snapshot.path.is_absolute() {
        return Err(invalid("PALS V6 snapshot root is not absolute"));
    }
    let held = snapshot
        .cwd
        .metadata()
        .map_err(|_| invalid("PALS V6 original snapshot root unavailable"))?;
    let capability = snapshot
        .directory
        .try_clone()
        .map_err(|_| invalid("PALS V6 snapshot root capability unavailable"))?
        .into_std_file()
        .metadata()
        .map_err(|_| invalid("PALS V6 snapshot root metadata unavailable"))?;
    let named = std::fs::symlink_metadata(snapshot.path)
        .map_err(|_| invalid("PALS V6 snapshot root pathname unavailable"))?;
    if !held.is_dir()
        || !capability.is_dir()
        || !named.is_dir()
        || named.file_type().is_symlink()
        || (held.dev(), held.ino()) != (capability.dev(), capability.ino())
        || (held.dev(), held.ino()) != (named.dev(), named.ino())
    {
        return Err(invalid("PALS V6 snapshot root differs from original owner"));
    }
    Ok(())
}

fn verify_named_pin(
    snapshot: &SelectedInputView<'_>,
    pin: &InputPin,
    relative: &str,
    executable: bool,
) -> Result<(), ArenaError> {
    if !rz_experiments::pals_private_snapshot_path_v5(relative)
        || pin.path != snapshot.path.join(relative)
    {
        return Err(invalid("PALS V6 input path differs from original snapshot"));
    }
    let (directory, name) = linux::input_parent(
        &snapshot.input_directory,
        &snapshot.input_subdirectories,
        Path::new(relative),
    )?;
    let mut options = OpenOptions::new();
    options
        .read(true)
        .follow(FollowSymlinks::No)
        .custom_flags(nix::libc::O_NOFOLLOW | nix::libc::O_NONBLOCK);
    let named = directory
        .open_with(name, &options)
        .map_err(|_| invalid("PALS V6 named input unavailable"))?
        .into_std();
    let named = named
        .metadata()
        .map_err(|_| invalid("PALS V6 named input metadata unavailable"))?;
    let held = pin
        .file
        .metadata()
        .map_err(|_| invalid("PALS V6 original input metadata unavailable"))?;
    if !held.is_file()
        || !named.is_file()
        || held.len() != pin.artifact.bytes
        || held.mode() & 0o222 != 0
        || held.nlink() != 1
        || (executable && held.mode() & 0o111 == 0)
        || (held.dev(), held.ino(), held.uid(), held.mode(), held.len())
            != (
                named.dev(),
                named.ino(),
                named.uid(),
                named.mode(),
                named.len(),
            )
    {
        return Err(invalid("PALS V6 named input differs from original pin"));
    }
    Ok(())
}

fn unchanged(before: &Metadata, after: &Metadata) -> bool {
    (
        before.dev(),
        before.ino(),
        before.len(),
        before.mtime(),
        before.mtime_nsec(),
        before.ctime(),
        before.ctime_nsec(),
    ) == (
        after.dev(),
        after.ino(),
        after.len(),
        after.mtime(),
        after.mtime_nsec(),
        after.ctime(),
        after.ctime_nsec(),
    )
}

fn verify_held_bytes(file: &File, expected: &[u8]) -> Result<(), ArenaError> {
    let before = file
        .metadata()
        .map_err(|_| invalid("PALS V6 original lock metadata unavailable"))?;
    if before.len() != expected.len() as u64 {
        return Err(invalid("PALS V6 original lock length differs"));
    }
    let mut offset = 0usize;
    let mut buffer = [0u8; 8192];
    while offset < expected.len() {
        let amount = buffer.len().min(expected.len() - offset);
        let read = match file.read_at(&mut buffer[..amount], offset as u64) {
            Ok(read) => read,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => return Err(invalid("PALS V6 original lock read failed")),
        };
        if read == 0 || buffer[..read] != expected[offset..offset + read] {
            return Err(invalid("PALS V6 original lock bytes differ"));
        }
        offset += read;
    }
    let after = file
        .metadata()
        .map_err(|_| invalid("PALS V6 original lock metadata unavailable"))?;
    if !unchanged(&before, &after) {
        return Err(invalid("PALS V6 original lock changed during read"));
    }
    Ok(())
}
