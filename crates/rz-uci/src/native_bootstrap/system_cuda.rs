//! Original LC0 system-CUDA startup custody, before product integration.
//!
//! This owner alone is not a supervisor, a device fence, or permission to
//! terminate the process. A caller must retain it together with the original
//! authenticated parent and must not replace an unknown physical result with
//! process exit. The legacy CLI intentionally rejects this path until that
//! caller and the normal shutdown/receipt path are connected.

use crate::pals_adaptive_bootstrap::runtime_assets::system_cuda::SystemCudaStartup;
use crate::rovent_bootstrap::launch::ConnectedParent;
use rz_eval::{
    asset::{AssetMetadata, MaiaAsset},
    error::{BackendError, FailureKind, FailureStage},
    onnx::{
        BackendConfig, Lc0StartupCustody, Lc0StartupState, OnnxBackend, OrtRuntime, Provider,
        SystemRuntime,
    },
};
use rz_experiments::{NativeRuntimeSourceV6, SystemRuntimeLockV6};
use std::{
    io,
    os::unix::net::UnixStream,
    sync::{Arc, atomic::AtomicBool},
    time::Instant,
};

/// A parent PID captured before the caller publishes any child endpoint. The
/// accepted stream is bound to this exact parent and the V6 runtime lock;
/// admission is neither device ownership nor a physical completion witness.
pub(super) struct Lc0PendingParent {
    original_parent_pid: i32,
    process_epoch: u64,
    runtime_lock_sha256: String,
    cuda_ordinal: i32,
}

impl Lc0PendingParent {
    pub(super) fn capture(
        process_epoch: u64,
        lock: &SystemRuntimeLockV6,
    ) -> Result<Self, BackendError> {
        lock.verify().map_err(|_| {
            BackendError::new(
                FailureKind::IdentityMismatch,
                FailureStage::Admission,
                "LC0 system CUDA lock is invalid",
            )
        })?;
        let spec = match &lock.runtime {
            NativeRuntimeSourceV6::SystemCuda(spec) if process_epoch != 0 => spec,
            _ => {
                return Err(BackendError::new(
                    FailureKind::InvalidInput,
                    FailureStage::Admission,
                    "LC0 parent admission requires a system lock and process epoch",
                ));
            }
        };
        let original_parent_pid = ConnectedParent::capture_original_parent().map_err(|_| {
            BackendError::new(
                FailureKind::BackendUnavailable,
                FailureStage::Admission,
                "LC0 original supervisor PID unavailable",
            )
        })?;
        Ok(Self {
            original_parent_pid,
            process_epoch,
            runtime_lock_sha256: lock.canonical_sha256.clone(),
            cuda_ordinal: spec.cuda_device_id,
        })
    }

    pub(super) fn authenticate(
        self,
        stream: UnixStream,
        until: Instant,
    ) -> io::Result<Arc<ConnectedParent>> {
        ConnectedParent::authenticate(
            stream,
            std::process::id(),
            self.original_parent_pid,
            self.process_epoch,
            &self.runtime_lock_sha256,
            self.cuda_ordinal,
            until,
        )
        .map(Arc::new)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Lc0SystemStartupPhase {
    BeforeOrtCall,
    /// Includes a reused process latch: absence of a new entry callback does
    /// not prove that previous native work never entered or has completed.
    OrtCallUnknown,
    OrtReady,
    SessionEntryUnknown,
    BackendReady,
}

fn matches_registered_provider(provider: Provider, device_id: i32, arena_bytes: u64) -> bool {
    matches!(
        provider,
        Provider::Cuda { device_id: actual, arena_bytes: arena }
            if actual == device_id && u64::try_from(arena) == Ok(arena_bytes)
    )
}

/// One original owner must be registered with the live supervisor before
/// calling `load`. No state in this object establishes physical completion.
pub(super) struct Lc0SystemStartupCustody {
    asset: Option<MaiaAsset>,
    system: Option<SystemRuntime>,
    runtime: Option<OrtRuntime>,
    startup: Option<SystemCudaStartup>,
    model: Option<Lc0StartupCustody>,
    ready: Option<Lc0ReadyObjects>,
    parent: Option<Arc<ConnectedParent>>,
    process_epoch: u64,
    phase: Lc0SystemStartupPhase,
    ort_entry_observed: bool,
    attempted: bool,
}

/// Same-owner intermediate state. A retained worker must consume the actual
/// backend through a recovery reservation; borrowing these fields is not a
/// physical completion witness or a successful worker handoff.
struct Lc0ReadyObjects {
    metadata: AssetMetadata,
    backend: Option<OnnxBackend>,
}

impl Lc0SystemStartupCustody {
    /// LC0's own supervisor must authenticate directly against the V6 system
    /// runtime lock digest. A Rovent V3 parent bound to an outer input lock is
    /// a different authority and cannot be silently substituted here.
    pub(super) fn new(
        asset: MaiaAsset,
        parent: Arc<ConnectedParent>,
        process_epoch: u64,
        lock: &SystemRuntimeLockV6,
    ) -> Result<Self, BackendError> {
        lock.verify().map_err(|_| {
            BackendError::new(
                FailureKind::IdentityMismatch,
                FailureStage::Admission,
                "LC0 system CUDA lock is invalid",
            )
        })?;
        let spec = match &lock.runtime {
            NativeRuntimeSourceV6::SystemCuda(spec) => spec,
            NativeRuntimeSourceV6::Cpu => {
                return Err(BackendError::new(
                    FailureKind::InvalidInput,
                    FailureStage::Admission,
                    "LC0 system CUDA requires a system CUDA runtime lock",
                ));
            }
        };
        if process_epoch == 0
            || !parent.matches(process_epoch, &lock.canonical_sha256, spec.cuda_device_id)
        {
            return Err(BackendError::new(
                FailureKind::IdentityMismatch,
                FailureStage::Admission,
                "LC0 supervisor is not bound to the V6 system runtime lock",
            ));
        }
        Ok(Self {
            asset: Some(asset),
            system: None,
            runtime: None,
            startup: None,
            model: None,
            ready: None,
            parent: Some(parent),
            process_epoch,
            phase: Lc0SystemStartupPhase::BeforeOrtCall,
            ort_entry_observed: false,
            attempted: false,
        })
    }

    pub(super) fn phase(&self) -> Lc0SystemStartupPhase {
        self.phase
    }

    pub(super) fn ort_entry_observed(&self) -> bool {
        self.ort_entry_observed
    }

    /// The already-owned authenticated parent channel is checked immediately
    /// before each native boundary; it is not a physical completion fence.
    /// Errors and unwinds leave all returned native objects in this owner.
    pub(super) fn load(
        &mut self,
        lock: &SystemRuntimeLockV6,
        until: Instant,
        cancelled: &AtomicBool,
        config: BackendConfig,
    ) -> Result<(), BackendError> {
        if self.attempted {
            return Err(BackendError::new(
                FailureKind::InvalidInput,
                FailureStage::Admission,
                "LC0 system startup owner already used",
            ));
        }
        self.attempted = true;
        let spec = match &lock.runtime {
            NativeRuntimeSourceV6::SystemCuda(spec) => spec,
            NativeRuntimeSourceV6::Cpu => {
                return Err(BackendError::new(
                    FailureKind::InvalidInput,
                    FailureStage::Admission,
                    "LC0 system CUDA requires a system CUDA runtime lock",
                ));
            }
        };
        if !matches_registered_provider(config.provider, spec.cuda_device_id, spec.cuda_arena_bytes)
        {
            return Err(BackendError::new(
                FailureKind::IdentityMismatch,
                FailureStage::Admission,
                "LC0 backend provider, device or arena differs from the system lock",
            ));
        }
        let parent = self.parent.as_ref().expect("original parent retained");
        if self.process_epoch == 0
            || !parent.matches(
                self.process_epoch,
                &lock.canonical_sha256,
                spec.cuda_device_id,
            )
        {
            return Err(BackendError::new(
                FailureKind::IdentityMismatch,
                FailureStage::Admission,
                "LC0 original supervisor and system CUDA lock differ",
            ));
        }
        let check_parent = || {
            parent.ensure_live().map_err(|error| {
                let detail = match error.kind() {
                    std::io::ErrorKind::PermissionDenied => "LC0 supervisor identity changed",
                    std::io::ErrorKind::UnexpectedEof => "LC0 supervisor disconnected",
                    std::io::ErrorKind::InvalidData => "LC0 supervisor channel had unexpected data",
                    _ => "LC0 supervisor channel I/O failed",
                };
                BackendError::new(
                    FailureKind::BackendUnavailable,
                    FailureStage::Admission,
                    detail,
                )
            })
        };
        let system_slot = &mut self.system;
        let runtime_slot = &mut self.runtime;
        let phase = &mut self.phase;
        let entry_observed = &mut self.ort_entry_observed;
        let startup =
            SystemCudaStartup::capture_and_load_until_with(lock, until, cancelled, |system| {
                *system_slot = Some(system.clone());
                check_parent()?;
                *phase = Lc0SystemStartupPhase::OrtCallUnknown;
                let mut rejected_before_entry = false;
                let runtime = OrtRuntime::load_system_cuda_with_checked_entry(system, || {
                    if let Err(error) = check_parent() {
                        rejected_before_entry = true;
                        return Err(error);
                    }
                    *entry_observed = true;
                    Ok(())
                });
                if rejected_before_entry {
                    // The checked callback ran on a new latch and rejected
                    // before native entry. This says nothing about work from
                    // other owners or earlier processes.
                    *phase = Lc0SystemStartupPhase::BeforeOrtCall;
                }
                let runtime = runtime?;
                // Save before the outer helper's identity/deadline checks.
                *runtime_slot = Some(runtime.clone());
                Ok(runtime)
            })?;
        self.startup = Some(startup);
        self.phase = Lc0SystemStartupPhase::OrtReady;
        let runtime = self.runtime.as_ref().expect("captured ORT runtime").clone();
        let asset = self.asset.take().expect("original LC0 asset");
        self.model = Some(Lc0StartupCustody::new(runtime, asset));
        let phase = &mut self.phase;
        self.model
            .as_mut()
            .expect("original LC0 startup owner")
            .load(config, || {
                check_parent()?;
                *phase = Lc0SystemStartupPhase::SessionEntryUnknown;
                Ok(())
            })?;
        self.phase = Lc0SystemStartupPhase::BackendReady;
        Ok(())
    }

    pub(super) fn model_state(&self) -> Option<Lc0StartupState> {
        self.model.as_ref().map(Lc0StartupCustody::state)
    }

    /// Move the original pair only after a successful startup. Both remain in
    /// this owner if later projection or worker admission fails.
    pub(super) fn prepare_handoff(&mut self) -> Result<(), BackendError> {
        if self.phase != Lc0SystemStartupPhase::BackendReady || self.ready.is_some() {
            return Err(BackendError::new(
                FailureKind::InvalidInput,
                FailureStage::Admission,
                "LC0 system startup is not ready for the original handoff",
            ));
        }
        let (metadata, backend) = self
            .model
            .as_mut()
            .and_then(Lc0StartupCustody::take_ready)
            .ok_or_else(|| {
                BackendError::new(
                    FailureKind::BackendUnavailable,
                    FailureStage::Admission,
                    "LC0 original model and backend are not both ready",
                )
            })?;
        self.ready = Some(Lc0ReadyObjects {
            metadata,
            backend: Some(backend),
        });
        Ok(())
    }

    /// Projection can inspect the original pair without gaining the ability
    /// to take and accidentally drop the physical backend. Worker transfer
    /// needs a separate reservation-bound operation on this owner.
    pub(super) fn ready_parts(&self) -> Option<(&AssetMetadata, &OnnxBackend)> {
        self.ready.as_ref().and_then(|ready| {
            ready
                .backend
                .as_ref()
                .map(|backend| (&ready.metadata, backend))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_must_match_original_system_device_and_arena() {
        let actual = Provider::Cuda {
            device_id: 1,
            arena_bytes: 4096,
        };
        assert!(matches_registered_provider(actual, 1, 4096));
        assert!(!matches_registered_provider(actual, 0, 4096));
        assert!(!matches_registered_provider(actual, 1, 8192));
        assert!(!matches_registered_provider(Provider::Cpu, 1, 4096));
    }
}

impl Drop for Lc0SystemStartupCustody {
    fn drop(&mut self) {
        if self.phase != Lc0SystemStartupPhase::BeforeOrtCall {
            // Defensive in-process quarantine only: the real caller still
            // needs to retain the process and original parent on unknown.
            std::mem::forget((
                self.asset.take(),
                self.system.take(),
                self.runtime.take(),
                self.startup.take(),
                self.model.take(),
                self.ready.take(),
                self.parent.take(),
            ));
        }
    }
}
