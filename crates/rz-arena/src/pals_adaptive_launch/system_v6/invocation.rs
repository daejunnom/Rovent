//! Late-bound V6 Fastchess arguments and their original abstract listeners.
//! Addresses identify transports, not role, game, peer or GPU completion.

use super::{
    LockedPalsSystemArenaPairV6, PalsSystemProxyInputsV1, PalsV6ExecutionStart,
    endpoint_snapshot::PalsV6EndpointSnapshot,
};
use crate::{
    ArenaError, NativeLaunchOwner,
    fastchess::FastchessInvocation,
    native_launch::linux::{self, V6EngineInvocation},
};
use nix::sys::stat::fstat;
use rz_experiments::NativeEngineRole;
use std::{
    ffi::OsString,
    os::{
        linux::net::SocketAddrExt,
        unix::{
            fs::MetadataExt,
            net::{SocketAddr, UnixListener},
        },
    },
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

static NEXT_SOCKET_SET: AtomicU64 = AtomicU64::new(0);

struct SocketOwner {
    name: String,
    listener: Arc<UnixListener>,
    inode: u64,
}

impl SocketOwner {
    fn bind(name: String) -> Result<Self, ArenaError> {
        let address = SocketAddr::from_abstract_name(name.as_bytes())
            .map_err(|_| ArenaError::Invalid("invalid V6 abstract listener name".into()))?;
        let listener = UnixListener::bind_addr(&address)
            .map_err(|_| ArenaError::Io("V6 abstract listener bind failed".into()))?;
        listener
            .set_nonblocking(true)
            .map_err(|_| ArenaError::Io("V6 abstract listener nonblocking failed".into()))?;
        let inode = fstat(&listener)
            .map_err(|_| ArenaError::Io("V6 abstract listener inode unavailable".into()))?
            .st_ino;
        let result = Self {
            name,
            listener: Arc::new(listener),
            inode,
        };
        result.verify()?;
        Ok(result)
    }

    fn verify(&self) -> Result<(), ArenaError> {
        let address = self
            .listener
            .local_addr()
            .map_err(|_| ArenaError::Io("V6 abstract listener address unavailable".into()))?;
        let inode = fstat(self.listener.as_ref())
            .map_err(|_| ArenaError::Io("V6 abstract listener inode unavailable".into()))?
            .st_ino;
        if address.as_abstract_name() != Some(self.name.as_bytes()) || inode != self.inode {
            return Err(ArenaError::Integrity(
                "V6 original abstract listener changed".into(),
            ));
        }
        Ok(())
    }
}

struct RoleSockets {
    grant: SocketOwner,
    cancel: SocketOwner,
    // This is a name offered to the proxy, which alone creates/owns the
    // child listener. It is not an arena-held listener or a spawn proof.
    child_name: String,
}

/// Must be retained with the original process/control owner after spawn,
/// including physical-unknown quarantine. Dropping before spawn closes only
/// these actual kernel sockets and never deletes a filesystem path.
pub(crate) struct PalsV6RoleSockets {
    execution: Arc<PalsV6ExecutionStart>,
    pair_sha256: String,
    sidecar: PalsSystemProxyInputsV1,
    cwd_device: u64,
    cwd_inode: u64,
    roles: [RoleSockets; 2],
}

impl PalsV6RoleSockets {
    pub(crate) fn bind_original(
        owner: &NativeLaunchOwner<LockedPalsSystemArenaPairV6>,
        execution: &Arc<PalsV6ExecutionStart>,
    ) -> Result<Arc<Self>, ArenaError> {
        crate::pals_adaptive_launch::verify_original_proxy_snapshots(owner)?;
        if execution.pair_sha256() != owner.spec().sha256() {
            return Err(ArenaError::Integrity(
                "V6 socket and execution pairs differ".into(),
            ));
        }
        let sidecar = owner
            .spec()
            .proxy_inputs()
            .ok_or_else(|| ArenaError::Invalid("V6 original proxy sidecar missing".into()))?
            .clone();
        let cwd = owner
            .snapshot
            .cwd
            .metadata()
            .map_err(|_| ArenaError::Io("V6 original cwd metadata unavailable".into()))?;
        let sequence = NEXT_SOCKET_SET
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| ArenaError::Budget("V6 socket name sequence exhausted".into()))?;
        let prefix = format!("rz-v6-{}-{sequence}", std::process::id());
        let role = |label: &str| -> Result<RoleSockets, ArenaError> {
            Ok(RoleSockets {
                grant: SocketOwner::bind(format!("{prefix}-{label}-grant"))?,
                cancel: SocketOwner::bind(format!("{prefix}-{label}-cancel"))?,
                child_name: format!("{prefix}-{label}-child"),
            })
        };
        let result = Self {
            execution: Arc::clone(execution),
            pair_sha256: owner.spec().sha256().into(),
            sidecar,
            cwd_device: cwd.dev(),
            cwd_inode: cwd.ino(),
            roles: [role("baseline")?, role("candidate")?],
        };
        result.verify_original(owner, execution)?;
        Ok(Arc::new(result))
    }

    pub(crate) fn listener(&self, role: NativeEngineRole, cancel: bool) -> &Arc<UnixListener> {
        let selected = &self.roles[match role {
            NativeEngineRole::Baseline => 0,
            NativeEngineRole::Candidate => 1,
        }];
        if cancel {
            &selected.cancel.listener
        } else {
            &selected.grant.listener
        }
    }

    pub(crate) fn verify_original(
        &self,
        owner: &NativeLaunchOwner<LockedPalsSystemArenaPairV6>,
        execution: &Arc<PalsV6ExecutionStart>,
    ) -> Result<(), ArenaError> {
        if !Arc::ptr_eq(&self.execution, execution)
            || self.pair_sha256 != owner.spec().sha256()
            || owner.spec().proxy_inputs() != Some(&self.sidecar)
        {
            return Err(ArenaError::Integrity(
                "V6 original socket owner differs".into(),
            ));
        }
        let cwd = owner
            .snapshot
            .cwd
            .metadata()
            .map_err(|_| ArenaError::Io("V6 original cwd metadata unavailable".into()))?;
        if (cwd.dev(), cwd.ino()) != (self.cwd_device, self.cwd_inode) {
            return Err(ArenaError::Integrity(
                "V6 original socket cwd differs".into(),
            ));
        }
        for role in [NativeEngineRole::Baseline, NativeEngineRole::Candidate] {
            PalsV6EndpointSnapshot::from_original(owner, role)?.verify_unchanged()?;
        }
        self.verify_listeners()?;
        Ok(())
    }

    pub(crate) fn verify_listeners(&self) -> Result<(), ArenaError> {
        for role in &self.roles {
            role.grant.verify()?;
            role.cancel.verify()?;
        }
        Ok(())
    }

    pub(crate) fn build_invocation(
        &self,
        owner: &NativeLaunchOwner<LockedPalsSystemArenaPairV6>,
        execution: &Arc<PalsV6ExecutionStart>,
    ) -> Result<FastchessInvocation, ArenaError> {
        self.verify_original(owner, execution)?;
        let spec = owner.spec();
        let proxy = linux::pin(&owner.snapshot.pins, &self.sidecar.proxy_binary)?;
        let roots = [
            owner.snapshot.path.join("baseline-runtime"),
            owner.snapshot.path.join("candidate-runtime"),
        ];
        let make = |role: NativeEngineRole| -> Result<V6EngineInvocation<'_>, ArenaError> {
            let selected = &self.roles[match role {
                NativeEngineRole::Baseline => 0,
                NativeEngineRole::Candidate => 1,
            }];
            let endpoint = self.sidecar.endpoint(spec, role)?;
            let path = |relative: &str| -> Result<String, ArenaError> {
                owner
                    .snapshot
                    .path
                    .join(relative)
                    .to_str()
                    .map(str::to_owned)
                    .ok_or_else(|| ArenaError::Invalid("V6 input path is not UTF-8".into()))
            };
            let arguments: Vec<OsString> = vec![
                "--pals-system-proxy=true".into(),
                format!(
                    "--pals-system-child-binary={}",
                    path(endpoint.child_snapshot_path)?
                )
                .into(),
                format!(
                    "--pals-system-startup-lock={}",
                    path(endpoint.startup_snapshot_path)?
                )
                .into(),
                format!(
                    "--pals-system-startup-lock-sha256={}",
                    endpoint.startup_lock_sha256
                )
                .into(),
                format!(
                    "--pals-system-startup-ms={}",
                    spec.endpoint(role)
                        .startup_lock
                        .config
                        .native_options
                        .startup_ms
                )
                .into(),
                format!("--pals-system-parent-abstract={}", selected.child_name).into(),
                format!("--pals-arena-grant-socket=abstract:{}", selected.grant.name).into(),
                format!(
                    "--pals-arena-cancel-socket=abstract:{}",
                    selected.cancel.name
                )
                .into(),
            ];
            Ok(V6EngineInvocation {
                binary: proxy,
                arguments,
            })
        };
        let overrides = [
            make(NativeEngineRole::Baseline)?,
            make(NativeEngineRole::Candidate)?,
        ];
        let invocation = linux::build_invocation_with_v6_overrides(
            spec,
            &owner.snapshot.pins,
            &owner.snapshot.path,
            &roots,
            Some(&overrides),
        )?;
        self.verify_original(owner, execution)?;
        Ok(invocation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abstract_listener_collision_preserves_original_fd() {
        let sequence = NEXT_SOCKET_SET.fetch_add(1, Ordering::Relaxed);
        let name = format!("rz-v6-test-{}-{sequence}", std::process::id());
        let original = SocketOwner::bind(name.clone()).unwrap();
        assert!(SocketOwner::bind(name).is_err());
        original.verify().unwrap();
    }
}
