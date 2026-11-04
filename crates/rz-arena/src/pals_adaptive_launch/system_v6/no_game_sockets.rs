//! Selected-role transports for an independently prepared NoGame smoke.
//! A bound address does not authenticate a peer or prove a game or spawn.

use super::{
    NoGameExecutionStartV1, ProxyEntrySmokeSnapshotOwnerV1,
    endpoint_snapshot::PalsV6EndpointSnapshot,
    invocation::{SocketOwner, next_socket_prefix},
};
use crate::ArenaError;
use rz_experiments::NativeEngineRole;
use std::{
    os::unix::{fs::MetadataExt, net::UnixListener},
    sync::Arc,
};

/// Retain this owner with the original process, including unknown completion.
pub(crate) struct NoGameRoleSocketsV1 {
    execution: Arc<NoGameExecutionStartV1>,
    plan_sha256: String,
    role: NativeEngineRole,
    cwd_device: u64,
    cwd_inode: u64,
    grant: SocketOwner,
    cancel: SocketOwner,
    child_name: String,
}

impl NoGameRoleSocketsV1 {
    pub(crate) fn bind_original(
        owner: &ProxyEntrySmokeSnapshotOwnerV1,
        execution: &Arc<NoGameExecutionStartV1>,
    ) -> Result<Arc<Self>, ArenaError> {
        if !Arc::ptr_eq(&owner.execution, execution) || !execution.matches_plan(&owner.plan) {
            return Err(ArenaError::Integrity(
                "NoGame socket execution owner differs".into(),
            ));
        }
        execution.require_running()?;
        PalsV6EndpointSnapshot::from_smoke(owner, owner.role)?;
        let cwd = owner
            .cwd
            .metadata()
            .map_err(|_| ArenaError::Io("NoGame cwd metadata unavailable".into()))?;
        let prefix = next_socket_prefix()?;
        let result = Self {
            execution: Arc::clone(execution),
            plan_sha256: owner.plan.sha256().into(),
            role: owner.role,
            cwd_device: cwd.dev(),
            cwd_inode: cwd.ino(),
            grant: SocketOwner::bind(format!("{prefix}-smoke-grant"))?,
            cancel: SocketOwner::bind(format!("{prefix}-smoke-cancel"))?,
            child_name: format!("{prefix}-smoke-child"),
        };
        result.verify_original(owner, execution)?;
        Ok(Arc::new(result))
    }

    pub(crate) fn verify_original(
        &self,
        owner: &ProxyEntrySmokeSnapshotOwnerV1,
        execution: &Arc<NoGameExecutionStartV1>,
    ) -> Result<(), ArenaError> {
        if !Arc::ptr_eq(&self.execution, execution)
            || !Arc::ptr_eq(&owner.execution, execution)
            || self.plan_sha256 != owner.plan.sha256()
            || self.role != owner.role
            || !execution.matches_plan(&owner.plan)
        {
            return Err(ArenaError::Integrity(
                "NoGame original socket owner differs".into(),
            ));
        }
        let cwd = owner
            .cwd
            .metadata()
            .map_err(|_| ArenaError::Io("NoGame cwd metadata unavailable".into()))?;
        if (cwd.dev(), cwd.ino()) != (self.cwd_device, self.cwd_inode) {
            return Err(ArenaError::Integrity(
                "NoGame original socket cwd differs".into(),
            ));
        }
        PalsV6EndpointSnapshot::from_smoke(owner, self.role)?;
        self.grant.verify()?;
        self.cancel.verify()?;
        Ok(())
    }

    pub(crate) fn role(&self) -> NativeEngineRole {
        self.role
    }
    pub(crate) fn listener(&self, cancel: bool) -> &Arc<UnixListener> {
        if cancel {
            &self.cancel.listener
        } else {
            &self.grant.listener
        }
    }
    pub(crate) fn grant_name(&self) -> &str {
        &self.grant.name
    }
    pub(crate) fn cancel_name(&self) -> &str {
        &self.cancel.name
    }
    /// A declaration for the proxy, not an arena-bound child listener.
    pub(crate) fn child_name(&self) -> &str {
        &self.child_name
    }
}
