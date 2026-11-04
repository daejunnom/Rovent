//! PALS system-runtime declarations and bounded child receipt observations.
//!
//! V6 is separate from the V5 CUDA bundle format. Merely observing a child
//! report is not process ownership, provider readiness, or physical completion
//! evidence. The V6 launch owner is not connected yet.

use crate::ArenaError;
use rz_experiments::{PalsSystemStartupLockV6, SystemRuntimeReceiptV6};

#[cfg(target_os = "linux")]
mod cancel_sources;
#[cfg(target_os = "linux")]
mod endpoint_snapshot;
#[cfg(target_os = "linux")]
mod invocation;
#[cfg(target_os = "linux")]
mod no_game_sockets;
#[cfg(target_os = "linux")]
pub(crate) use cancel_sources::{CancelSourceObservation, compiled_cancel_source_observation};
#[cfg(target_os = "linux")]
pub(crate) use endpoint_snapshot::verify_original_proxy_snapshots;
#[cfg(target_os = "linux")]
pub(crate) use invocation::PalsV6RoleSockets;
#[cfg(target_os = "linux")]
pub(crate) use no_game_sockets::NoGameRoleSocketsV1;
#[cfg(target_os = "linux")]
pub(crate) use proxy_entry_smoke::{ProxyEntryFunctionalSmokeOwnerV1, SMOKE_RECEIPT_FILE};
#[cfg(target_os = "linux")]
mod deadline_control;
#[cfg(target_os = "linux")]
pub(crate) use deadline_control::{
    SmokeControlProgress, SmokeDeadlineControl, SmokeGrantObservation,
};
mod deadline_grant;
mod deadline_sources;
#[cfg(target_os = "linux")]
mod direct_smoke_launch;
#[cfg(target_os = "linux")]
pub(crate) use direct_smoke_launch::DirectSmokeBinding;
mod execution_start;
#[cfg(target_os = "linux")]
mod no_game_plan;
#[cfg(target_os = "linux")]
pub(crate) use no_game_plan::{
    LockedNoGameSmokePlanV1, NO_GAME_SMOKE_PLAN_DOMAIN, NoGameExecutionStartV1,
    NoGameSmokeBudgetV1, NoGameSmokeEndpointV1, NoGameSmokePlanV1,
};
mod pair;
mod pair_native;
#[cfg(target_os = "linux")]
mod preflight;
#[cfg(target_os = "linux")]
mod proxy_entry_smoke;
#[cfg(target_os = "linux")]
mod smoke_preparation;
#[cfg(target_os = "linux")]
pub(crate) use smoke_preparation::{
    ProxyEntrySmokeSnapshotOwnerV1, prepare_proxy_entry_smoke_snapshot_v1,
};
mod proxy_inputs;
pub use deadline_sources::*;
pub use execution_start::*;
pub use pair::*;
pub use proxy_inputs::*;

/// A verified startup declaration, not a running child or an admission token.
#[derive(Clone, Debug)]
pub struct PalsSystemArenaEndpointV6 {
    startup: PalsSystemStartupLockV6,
}

impl PalsSystemArenaEndpointV6 {
    pub fn new(startup: PalsSystemStartupLockV6) -> Result<Self, ArenaError> {
        startup.verify()?;
        Ok(Self { startup })
    }

    pub fn startup_lock(&self) -> &PalsSystemStartupLockV6 {
        &self.startup
    }

    /// A declaration alone must never be used as a launch capability. The
    /// native owner, source snapshots and session receipt consumer still need
    /// to be connected before this execution profile can be admitted.
    pub fn require_native_owner(&self) -> Result<(), ArenaError> {
        Err(ArenaError::Invalid(
            "PALS V6 system-runtime launch owner is not connected".into(),
        ))
    }

    /// Read the claimed receipts for an expected child from the role directory
    /// held by the caller. The original supervisor must separately bind that
    /// role directory and PID to its actual process and validate its exit.
    /// Unknown or malformed receipts are retained as observations, not success.
    #[cfg(target_os = "linux")]
    pub fn observe_child_receipts(
        &self,
        process_id: u32,
        role_directory: &cap_std::fs::Dir,
    ) -> Result<PalsSystemChildReceiptClaimsV6, ArenaError> {
        use cap_fs_ext::DirExt;

        self.startup.verify()?;
        if process_id == 0 {
            return Err(ArenaError::Invalid("PALS child PID must be nonzero".into()));
        }
        let session = match role_directory.open_dir_nofollow(format!("native-process-{process_id}"))
        {
            Ok(session) => session,
            Err(error) => {
                let unavailable = format!("PALS child receipt directory is unavailable: {error}");
                return Ok(PalsSystemChildReceiptClaimsV6 {
                    endpoint_id: self.startup.config.endpoint.id.clone(),
                    process_id,
                    process_generation: self.startup.config.process_generation,
                    startup: PalsSystemReceiptClaimV6::Unavailable {
                        reason: unavailable.clone(),
                    },
                    termination: PalsSystemReceiptClaimV6::Unavailable {
                        reason: unavailable,
                    },
                });
            }
        };
        let lock = &self.startup.config.system_runtime_lock;
        let generation = self.startup.config.process_generation;
        Ok(PalsSystemChildReceiptClaimsV6 {
            endpoint_id: self.startup.config.endpoint.id.clone(),
            process_id,
            process_generation: generation,
            startup: observe_receipt(
                &session,
                "system-runtime-startup.v6.json",
                lock,
                process_id,
                generation,
            ),
            termination: observe_receipt(
                &session,
                "system-runtime-termination.v6.json",
                lock,
                process_id,
                generation,
            ),
        })
    }

    /// Check both reports against the same declared child identity and system
    /// runtime. This validates the child's claims only: the native supervisor
    /// must independently bind `process_id` to the actual process and verify
    /// its exit and the PALS session's own physical-lifetime evidence.
    pub fn validate_reported_session(
        &self,
        claims: &PalsSystemChildReceiptClaimsV6,
    ) -> Result<PalsSystemReportedSessionV6, ArenaError> {
        use rz_experiments::{NativeRuntimeSourceV6, PhysicalCompletionV6, ProviderReadinessV6};

        self.startup.verify()?;
        let lock = &self.startup.config.system_runtime_lock;
        let generation = self.startup.config.process_generation;
        if claims.endpoint_id != self.startup.config.endpoint.id
            || claims.process_id == 0
            || claims.process_generation != generation
        {
            return Err(ArenaError::Integrity(
                "PALS system session identity differs".into(),
            ));
        }
        let PalsSystemReceiptClaimV6::Reported {
            bytes: startup_bytes,
            receipt: startup,
        } = &claims.startup
        else {
            return Err(ArenaError::Integrity(
                "PALS system startup report is unavailable or invalid".into(),
            ));
        };
        let PalsSystemReceiptClaimV6::Reported {
            bytes: termination_bytes,
            receipt: termination,
        } = &claims.termination
        else {
            return Err(ArenaError::Integrity(
                "PALS system termination report is unavailable or invalid".into(),
            ));
        };
        let decode = |bytes: &[u8]| -> Result<SystemRuntimeReceiptV6, ArenaError> {
            if bytes.len() > 256 << 10 {
                return Err(ArenaError::Integrity(
                    "PALS system receipt exceeds its byte budget".into(),
                ));
            }
            let input = std::str::from_utf8(bytes)
                .map_err(|_| ArenaError::Integrity("PALS system receipt is not UTF-8".into()))?;
            Ok(SystemRuntimeReceiptV6::from_json(
                input,
                lock,
                claims.process_id,
                generation,
            )?)
        };
        if decode(startup_bytes)? != *startup || decode(termination_bytes)? != *termination {
            return Err(ArenaError::Integrity(
                "PALS system report differs from its preserved bytes".into(),
            ));
        }
        let NativeRuntimeSourceV6::SystemCuda(system) = &lock.runtime else {
            return Err(ArenaError::Integrity(
                "PALS V6 session is not system CUDA".into(),
            ));
        };
        if startup.provider_ready != ProviderReadinessV6::Ready
            || startup.physical_completion != PhysicalCompletionV6::Unknown
            || startup.observed_identity_sha256.as_deref()
                != Some(system.system_identity_sha256.as_str())
            || termination.provider_ready != ProviderReadinessV6::Ready
            || !matches!(
                termination.physical_completion,
                PhysicalCompletionV6::Confirmed { .. }
            )
            || termination.observed_identity_sha256.as_deref()
                != Some(system.system_identity_sha256.as_str())
        {
            return Err(ArenaError::Integrity(
                "PALS system startup observation is incomplete".into(),
            ));
        }
        for receipt in [startup, termination] {
            system.validate_observed(&receipt.observed_libraries)?;
            if system
                .libraries
                .iter()
                .filter(|library| library.required_mapping)
                .any(|library| {
                    !receipt.observed_libraries.iter().any(|observed| {
                        observed.role == library.role && observed.mapped == Some(true)
                    })
                })
            {
                return Err(ArenaError::Integrity(
                    "PALS required system library mapping was not observed".into(),
                ));
            }
        }
        let strict_inventory_empty = match (
            &startup.unregistered_native_mappings,
            &termination.unregistered_native_mappings,
        ) {
            (Some(start), Some(end)) => Some(start.is_empty() && end.is_empty()),
            (Some(mappings), _) | (_, Some(mappings)) if !mappings.is_empty() => Some(false),
            _ => None,
        };
        let optional_mappings_observed = [startup, termination].iter().all(|receipt| {
            receipt
                .observed_libraries
                .iter()
                .all(|library| library.mapped.is_some())
        });
        Ok(PalsSystemReportedSessionV6 {
            strict_inventory_empty,
            optional_mappings_observed,
        })
    }
}

/// Reported mapping completeness is distinct from the required registered
/// library observations. Additional or unobserved mappings do not establish
/// the stricter inventory claim, and do not themselves prove a conflict.
/// Neither this value nor a child's completion report authenticates the child.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PalsSystemReportedSessionV6 {
    pub strict_inventory_empty: Option<bool>,
    pub optional_mappings_observed: bool,
}

/// A structurally valid report is still an unauthenticated child claim. Raw
/// bytes remain available even when decoding or identity validation fails.
#[derive(Clone, Debug)]
pub enum PalsSystemReceiptClaimV6 {
    Unavailable {
        reason: String,
    },
    Invalid {
        bytes: Vec<u8>,
        reason: String,
    },
    Reported {
        bytes: Vec<u8>,
        receipt: SystemRuntimeReceiptV6,
    },
}

/// These are wire observations only; a parent process exit, provider readiness
/// and actual GPU completion must be established by the original owner.
#[derive(Clone, Debug)]
pub struct PalsSystemChildReceiptClaimsV6 {
    pub endpoint_id: String,
    pub process_id: u32,
    pub process_generation: u64,
    pub startup: PalsSystemReceiptClaimV6,
    pub termination: PalsSystemReceiptClaimV6,
}

#[cfg(target_os = "linux")]
fn observe_receipt(
    session: &cap_std::fs::Dir,
    name: &str,
    lock: &rz_experiments::SystemRuntimeLockV6,
    process_id: u32,
    generation: u64,
) -> PalsSystemReceiptClaimV6 {
    let bytes = match crate::native_runner::linux::read_file(session, name, 256 << 10) {
        Ok(bytes) => bytes,
        Err(error) => {
            return PalsSystemReceiptClaimV6::Unavailable {
                reason: error.to_string(),
            };
        }
    };
    let parsed = std::str::from_utf8(&bytes)
        .map_err(|_| "system runtime receipt is not UTF-8".to_owned())
        .and_then(|text| {
            SystemRuntimeReceiptV6::from_json(text, lock, process_id, generation)
                .map_err(|error| error.to_string())
        });
    match parsed {
        Ok(receipt) => PalsSystemReceiptClaimV6::Reported { bytes, receipt },
        Err(reason) => PalsSystemReceiptClaimV6::Invalid { bytes, reason },
    }
}
