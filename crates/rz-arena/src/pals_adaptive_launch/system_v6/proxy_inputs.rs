//! Additive pinned-input declaration for a future direct-parent PALS proxy.
//!
//! This does not spawn Fastchess or an engine, modify V6 canonical bytes, or
//! attest executable identity or physical completion. The native launch owner
//! must snapshot and verify these bytes, then bind each actual proxy PID and
//! its authenticated direct child to its own run and completion evidence.

use super::LockedPalsSystemArenaPairV6;
use crate::ArenaError;
use rz_experiments::{ArtifactRef, NativeEngineRole, pals_private_snapshot_path_v5};
use serde::{Deserialize, Serialize};

pub const PALS_SYSTEM_PROXY_INPUTS_V1_DOMAIN: &str = "rz-pals-system-proxy-inputs/1";

/// A separate sidecar leaves the historical V6 pair parser/digest unchanged.
/// The proxy executable is distinct from the child executable and must be
/// independently declared, snapshotted and checked by the original owner.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsSystemProxyInputsV1 {
    pub domain: String,
    pub pair_sha256: String,
    pub proxy_binary: ArtifactRef,
    pub proxy_snapshot_path: String,
}

/// A projection of declared paths only. The executor must resolve all paths
/// through original pinned capabilities; none of these strings authorizes
/// process spawning or establishes proxy/child PID identity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PalsSystemProxyEndpointV1<'a> {
    pub endpoint_id: &'a str,
    pub proxy_snapshot_path: &'a str,
    pub child_snapshot_path: &'a str,
    pub startup_snapshot_path: &'a str,
    pub startup_lock_sha256: &'a str,
    pub process_generation: u64,
    pub runtime_directory: &'a str,
}

impl PalsSystemProxyInputsV1 {
    pub fn validate_for_pair(&self, pair: &LockedPalsSystemArenaPairV6) -> Result<(), ArenaError> {
        let invalid = |detail: &str| ArenaError::Invalid(detail.into());
        if self.domain != PALS_SYSTEM_PROXY_INPUTS_V1_DOMAIN
            || self.pair_sha256 != pair.sha256()
            || !pals_private_snapshot_path_v5(&self.proxy_snapshot_path)
        {
            return Err(invalid("proxy sidecar identity or snapshot path differs"));
        }
        self.proxy_binary.validate()?;
        let mut source_already_registered = false;
        for existing in pair.input().declared_artifacts() {
            if existing.path == self.proxy_binary.path {
                if existing != &self.proxy_binary {
                    return Err(invalid("proxy source has contradictory artifact identity"));
                }
                if pair.snapshot_relative_path(existing).is_none() {
                    return Err(invalid(
                        "existing proxy source has no explicit V6 snapshot mapping",
                    ));
                }
                source_already_registered = true;
            }
            if let Some(path) = pair.snapshot_relative_path(existing) {
                if path == self.proxy_snapshot_path && existing != &self.proxy_binary {
                    return Err(invalid("proxy snapshot aliases another artifact"));
                }
                if existing == &self.proxy_binary && path != self.proxy_snapshot_path {
                    return Err(invalid("proxy source has a different registered snapshot"));
                }
            }
        }
        let extra = if source_already_registered {
            0
        } else {
            self.proxy_binary.bytes
        };
        let input = pair
            .input()
            .unique_input_bytes()?
            .checked_add(extra)
            .ok_or_else(|| invalid("proxy input bytes overflow"))?;
        if input > pair.input().budget.max_input_bytes {
            return Err(invalid("proxy binary exceeds the registered input budget"));
        }
        // Two proxies, their two children, and the runner may coexist. This
        // request is not an observed process count or a kernel enforcement.
        if pair.input().budget.max_child_processes < 5 {
            return Err(invalid("pair process budget excludes proxy children"));
        }
        Ok(())
    }

    pub fn endpoint<'a>(
        &'a self,
        pair: &'a LockedPalsSystemArenaPairV6,
        role: NativeEngineRole,
    ) -> Result<PalsSystemProxyEndpointV1<'a>, ArenaError> {
        self.validate_for_pair(pair)?;
        let endpoint = pair.endpoint(role);
        Ok(PalsSystemProxyEndpointV1 {
            endpoint_id: &endpoint.startup_lock.config.endpoint.id,
            proxy_snapshot_path: &self.proxy_snapshot_path,
            child_snapshot_path: &endpoint.binary_snapshot_path,
            startup_snapshot_path: &endpoint.startup_snapshot_path,
            startup_lock_sha256: &endpoint.startup_lock.canonical_sha256,
            process_generation: endpoint.startup_lock.config.process_generation,
            runtime_directory: &endpoint.startup_lock.config.runtime_directory,
        })
    }
}
