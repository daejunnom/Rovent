//! Original-owner binding for a direct proxy-entry functional smoke.
//!
//! This prepares only the common, role-specific launch inputs. It does not
//! encode a NoGame grant scope, spawn a process, or prove physical completion.
//! The caller retains all original owners, including the child custody slot.

use super::{
    LockedPalsSystemArenaPairV6, PalsV6ExecutionStart, endpoint_snapshot::PalsV6EndpointSnapshot,
    invocation::PalsV6RoleSockets, proxy_entry_smoke::ProxyEntryFunctionalSmokeOwnerV1,
};
use crate::{ArenaError, NativeLaunchOwner, process::SpawnCustody};
use rz_experiments::NativeEngineRole;
use std::{
    ffi::OsString,
    fs::File,
    os::{linux::net::SocketAddrExt, unix::fs::MetadataExt},
    sync::{Arc, Mutex, atomic::Ordering},
};

/// Borrows the actual launch owner and original child slot; the same smoke,
/// execution, role channels and duplicate of the pinned proxy FD are retained
/// together. A caller must retain the original custody after any later spawn.
pub(super) struct DirectSmokeBinding<'a> {
    owner: &'a NativeLaunchOwner<LockedPalsSystemArenaPairV6>,
    smoke: &'a Arc<Mutex<ProxyEntryFunctionalSmokeOwnerV1>>,
    execution: &'a Arc<PalsV6ExecutionStart>,
    sockets: &'a Arc<PalsV6RoleSockets>,
    custody: &'a mut SpawnCustody,
    proxy: Arc<File>,
    role: NativeEngineRole,
    owner_id: [u8; 16],
    owner_epoch: u64,
    common_arguments: Vec<OsString>,
}

impl<'a> DirectSmokeBinding<'a> {
    /// All inputs are borrowed so an error leaves the original owner, socket
    /// and custody with the caller. The cloned FD references the same open
    /// proxy file; it is not reopened by pathname.
    pub(super) fn bind(
        owner: &'a NativeLaunchOwner<LockedPalsSystemArenaPairV6>,
        smoke: &'a Arc<Mutex<ProxyEntryFunctionalSmokeOwnerV1>>,
        execution: &'a Arc<PalsV6ExecutionStart>,
        sockets: &'a Arc<PalsV6RoleSockets>,
        custody: &'a mut SpawnCustody,
        role: NativeEngineRole,
        owner_id: [u8; 16],
        owner_epoch: u64,
    ) -> Result<Self, ArenaError> {
        if custody.child().is_some() || custody.pending.load(Ordering::Acquire) {
            return Err(ArenaError::Invalid(
                "direct smoke custody is already in use".into(),
            ));
        }
        {
            let selected = smoke.try_lock().map_err(|_| {
                ArenaError::Invalid("direct smoke original owner unavailable".into())
            })?;
            selected.verify_original_binding(owner, execution, role, owner_id, owner_epoch)?;
        }
        sockets.verify_original(owner, execution)?;
        let endpoint = PalsV6EndpointSnapshot::from_original(owner, role)?;
        let proxy = Arc::new(endpoint.proxy_file().try_clone().map_err(|_| {
            ArenaError::Io("direct smoke original proxy FD duplicate failed".into())
        })?);
        let pair = owner.spec();
        let sidecar = pair.proxy_inputs().ok_or_else(|| {
            ArenaError::Invalid("direct smoke original proxy sidecar missing".into())
        })?;
        let selected = sidecar.endpoint(pair, role)?;
        let path = |relative: &str| -> Result<String, ArenaError> {
            owner
                .snapshot
                .path
                .join(relative)
                .to_str()
                .map(str::to_owned)
                .ok_or_else(|| ArenaError::Invalid("direct smoke input path is not UTF-8".into()))
        };
        let address = |cancel| -> Result<String, ArenaError> {
            let address = sockets.listener(role, cancel).local_addr().map_err(|_| {
                ArenaError::Io("direct smoke role listener address unavailable".into())
            })?;
            let name = address.as_abstract_name().ok_or_else(|| {
                ArenaError::Invalid("direct smoke requires an original abstract listener".into())
            })?;
            let name = std::str::from_utf8(name).map_err(|_| {
                ArenaError::Invalid("direct smoke listener name is not UTF-8".into())
            })?;
            if name.is_empty() || !name.is_ascii() || name.chars().any(char::is_control) {
                return Err(ArenaError::Invalid(
                    "direct smoke listener name invalid".into(),
                ));
            }
            Ok(format!("abstract:{name}"))
        };
        let common_arguments = vec![
            "--pals-system-proxy=true".into(),
            "--pals-proxy-entry-functional-smoke=true".into(),
            format!(
                "--pals-system-child-binary={}",
                path(selected.child_snapshot_path)?
            )
            .into(),
            format!(
                "--pals-system-startup-lock={}",
                path(selected.startup_snapshot_path)?
            )
            .into(),
            format!(
                "--pals-system-startup-lock-sha256={}",
                selected.startup_lock_sha256
            )
            .into(),
            format!(
                "--pals-system-startup-ms={}",
                pair.endpoint(role)
                    .startup_lock
                    .config
                    .native_options
                    .startup_ms
            )
            .into(),
            format!(
                "--pals-system-parent-abstract={}",
                sockets.selected_child_name(role)
            )
            .into(),
            format!("--pals-arena-grant-socket={}", address(false)?).into(),
            format!("--pals-arena-cancel-socket={}", address(true)?).into(),
        ];
        let binding = Self {
            owner,
            smoke,
            execution,
            sockets,
            custody,
            proxy,
            role,
            owner_id,
            owner_epoch,
            common_arguments,
        };
        binding.verify_original()?;
        Ok(binding)
    }

    /// Recheck immediately before a later controlled spawn. This is not a
    /// process identity, physical completion or permission to invent scope.
    pub(super) fn verify_original(&self) -> Result<(), ArenaError> {
        self.execution.require_running()?;
        self.smoke
            .try_lock()
            .map_err(|_| ArenaError::Invalid("direct smoke original owner unavailable".into()))?
            .verify_original_binding(
                self.owner,
                self.execution,
                self.role,
                self.owner_id,
                self.owner_epoch,
            )?;
        self.sockets.verify_original(self.owner, self.execution)?;
        let endpoint = PalsV6EndpointSnapshot::from_original(self.owner, self.role)?;
        let held = endpoint.proxy_file().metadata().map_err(|_| {
            ArenaError::Io("direct smoke original proxy metadata unavailable".into())
        })?;
        let cloned = self.proxy.metadata().map_err(|_| {
            ArenaError::Io("direct smoke duplicate proxy metadata unavailable".into())
        })?;
        if (held.dev(), held.ino()) != (cloned.dev(), cloned.ino()) {
            return Err(ArenaError::Integrity(
                "direct smoke original proxy FD changed".into(),
            ));
        }
        Ok(())
    }

    pub(super) fn proxy(&self) -> &Arc<File> {
        &self.proxy
    }
    pub(super) fn common_arguments(&self) -> &[OsString] {
        &self.common_arguments
    }
    pub(super) fn custody(&mut self) -> &mut SpawnCustody {
        self.custody
    }
}
