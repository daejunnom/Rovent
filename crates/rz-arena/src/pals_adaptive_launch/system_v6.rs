//! PALS system-runtime declarations and bounded child receipt observations.
//!
//! V6 is separate from the V5 CUDA bundle format. Merely observing a child
//! report is not process ownership, provider readiness, or physical completion
//! evidence. The V6 launch owner is not connected yet.

use crate::ArenaError;
use rz_experiments::{PalsSystemStartupLockV6, SystemRuntimeReceiptV6};

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
        let session = role_directory
            .open_dir_nofollow(format!("native-process-{process_id}"))
            .map_err(|_| ArenaError::Io("PALS child receipt directory is unavailable".into()))?;
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
