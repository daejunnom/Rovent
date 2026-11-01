//! System-CUDA startup boundary for the separate V6 runtime lock.
//!
//! This module never copies native libraries, infers CUDA from an environment
//! variable, or interprets historical V5 bundle locks as system-runtime locks.
//! The caller must authenticate the original V6 lock and its execution owner.
//! ORT initialization, provider readiness, inference and physical completion
//! are separate observations and are never promoted by this module.

use rz_eval::error::{BackendError, CauseCode, FailureKind, FailureStage};
use rz_eval::onnx::{
    OrtRuntime, SystemCaptureLimits, SystemLibraryRequest, SystemMappingAvailability, SystemRuntime,
};
use rz_experiments::{
    NativeRuntimeSourceV6, SystemLibraryObservationV6, SystemPlatformV6, SystemRuntimeLockV6,
    SystemRuntimeSpecV6,
};
use std::{
    path::PathBuf,
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

fn invalid(detail: &'static str) -> BackendError {
    BackendError::new(FailureKind::InvalidInput, FailureStage::Admission, detail)
}

fn check_cancellation(cancelled: &AtomicBool) -> Result<(), BackendError> {
    if cancelled.load(Ordering::Acquire) {
        return Err(BackendError::new(
            FailureKind::BackendUnavailable,
            FailureStage::Admission,
            "Rovent startup was canceled",
        ));
    }
    Ok(())
}

fn identity_error(detail: &'static str) -> BackendError {
    BackendError::new(
        FailureKind::IdentityMismatch,
        FailureStage::Admission,
        detail,
    )
}

fn hex(bytes: &[u8; 32]) -> String {
    use std::fmt::Write;
    let mut value = String::with_capacity(64);
    for byte in bytes {
        // Formatting to a String cannot fail.
        write!(&mut value, "{byte:02x}").expect("format byte into String");
    }
    value
}

fn system_spec(lock: &SystemRuntimeLockV6) -> Result<&SystemRuntimeSpecV6, BackendError> {
    lock.verify()
        .map_err(|_| invalid("invalid V6 system runtime lock"))?;
    match &lock.runtime {
        NativeRuntimeSourceV6::SystemCuda(spec) if spec.platform == SystemPlatformV6::Linux => {
            Ok(spec)
        }
        NativeRuntimeSourceV6::SystemCuda(_) => Err(invalid("system CUDA profile is not Linux")),
        NativeRuntimeSourceV6::Cpu => Err(invalid("CPU runtime cannot select system CUDA")),
    }
}

fn library_observations(
    owner: &SystemRuntime,
) -> Result<Vec<SystemLibraryObservationV6>, BackendError> {
    let mapping = owner.observe_mappings().map_err(|error| {
        BackendError::new(
            FailureKind::BackendUnavailable,
            FailureStage::Backend,
            "cannot observe system CUDA runtime mappings",
        )
        .with_external_cause(CauseCode::RuntimeLibraryLoad, &error)
    })?;
    if !mapping.unregistered_native_mappings.is_empty() {
        return Err(identity_error("unregistered system native library mapping"));
    }
    owner
        .libraries()
        .into_iter()
        .map(|item| {
            let canonical_path = item
                .canonical_path
                .to_str()
                .ok_or_else(|| invalid("system runtime canonical path is not UTF-8"))?
                .to_owned();
            Ok(SystemLibraryObservationV6 {
                mapped: if mapping.availability == SystemMappingAvailability::Observed
                    && mapping.preexisting_mapping_observed
                {
                    Some(mapping.mapped_roles.iter().any(|role| role == &item.role))
                } else {
                    None
                },
                role: item.role,
                canonical_path,
                sha256: hex(&item.sha256),
                bytes: item.bytes,
                device: item.device,
                inode: item.inode,
            })
        })
        .collect()
}

/// Captured system CUDA runtime whose owner and exact V6 registration survive
/// through handoff to the native backend. Provider readiness and physical GPU
/// completion are not asserted by construction.
pub struct SystemCudaStartup {
    runtime: OrtRuntime,
    spec: SystemRuntimeSpecV6,
}

impl SystemCudaStartup {
    pub fn capture_and_load_until(
        lock: &SystemRuntimeLockV6,
        until: Instant,
        cancelled: &AtomicBool,
    ) -> Result<Self, BackendError> {
        let spec = system_spec(lock)?;
        check_cancellation(cancelled)?;
        let remaining = until
            .checked_duration_since(Instant::now())
            .filter(|duration| !duration.is_zero())
            .ok_or_else(|| invalid("system CUDA startup deadline expired"))?;
        let requests: Vec<_> = spec
            .libraries
            .iter()
            .map(|item| SystemLibraryRequest {
                role: item.role.clone(),
                path: PathBuf::from(&item.requested_path),
                required_mapping: item.required_mapping,
            })
            .collect();
        let limits = SystemCaptureLimits {
            max_libraries: usize::try_from(spec.capture_max_libraries)
                .map_err(|_| invalid("capture library budget exceeds address space"))?,
            max_file_bytes: spec.capture_max_file_bytes,
            max_total_bytes: spec.capture_max_total_bytes,
            max_elapsed: Duration::from_millis(spec.capture_timeout_ms).min(remaining),
        };
        let owner = SystemRuntime::capture(requests, limits).map_err(|error| {
            BackendError::new(
                FailureKind::BackendUnavailable,
                FailureStage::Backend,
                "system CUDA native library capture failed",
            )
            .with_external_cause(CauseCode::RuntimeLibraryLoad, &error)
        })?;
        check_cancellation(cancelled)?;
        if Instant::now() >= until {
            return Err(BackendError::new(
                FailureKind::BackendUnavailable,
                FailureStage::Backend,
                "system CUDA startup deadline expired before runtime initialization",
            ));
        }
        if hex(&owner.identity_sha256()) != spec.system_identity_sha256 {
            return Err(identity_error(
                "captured system runtime identity differs from lock",
            ));
        }
        // A capture does not map every lazy library. Check only identity here.
        spec.validate_observed(&library_observations(&owner)?)
            .map_err(|_| identity_error("captured system library differs from lock"))?;
        check_cancellation(cancelled)?;
        if Instant::now() >= until {
            return Err(BackendError::new(
                FailureKind::BackendUnavailable,
                FailureStage::Backend,
                "system CUDA startup deadline expired before runtime initialization",
            ));
        }
        // ORT retains the owner even if cancellation arrives while its native
        // initializer is running; no provider or physical completion is inferred.
        let runtime = OrtRuntime::load_system_cuda(&owner)?;
        check_cancellation(cancelled)?;
        if !runtime.is_cuda() || runtime.cuda_runtime_identity() != Some(owner.identity_sha256()) {
            return Err(identity_error(
                "loaded runtime differs from system CUDA selection",
            ));
        }
        if Instant::now() >= until {
            return Err(BackendError::new(
                FailureKind::BackendUnavailable,
                FailureStage::Backend,
                "system CUDA startup deadline expired after runtime initialization",
            ));
        }
        Ok(Self {
            runtime,
            spec: spec.clone(),
        })
    }

    /// Pass the actual owning ORT runtime to a backend factory; the returned
    /// clone retains the native library owner. This is not provider readiness.
    pub fn runtime(&self) -> OrtRuntime {
        self.runtime.clone()
    }

    /// Capture current child mappings against the frozen registration. This
    /// never infers provider readiness or physical GPU completion.
    pub fn observe(&self) -> Result<Vec<SystemLibraryObservationV6>, BackendError> {
        let owner = self
            .runtime
            .system_runtime()
            .ok_or_else(|| identity_error("system CUDA owner disappeared"))?;
        let observed = library_observations(owner)?;
        self.spec
            .validate_observed(&observed)
            .map_err(|_| identity_error("system library observation differs from lock"))?;
        Ok(observed)
    }

    /// Verify required mappings after the actual provider setup. Call only at
    /// that point: lazy providers are not expected to map during capture.
    pub fn verify_ready_mappings(&self) -> Result<Vec<SystemLibraryObservationV6>, BackendError> {
        let owner = self
            .runtime
            .system_runtime()
            .ok_or_else(|| identity_error("system CUDA owner disappeared"))?;
        owner.verify_mappings().map_err(|error| {
            BackendError::new(
                FailureKind::IdentityMismatch,
                FailureStage::Backend,
                "system CUDA required mappings not confirmed",
            )
            .with_external_cause(CauseCode::RuntimeLibraryLoad, &error)
        })?;
        let observed = self.observe()?;
        self.spec
            .validate_ready_observed(&observed)
            .map_err(|_| identity_error("system CUDA registered mappings not ready"))?;
        Ok(observed)
    }

    /// Normal engine execution may proceed with explicitly unknown mapping
    /// evidence. Observed conflicts remain errors; an acceptance caller must
    /// use `verify_ready_mappings` and retain the original physical fence.
    pub fn observe_ready_mappings(&self) -> Result<Vec<SystemLibraryObservationV6>, BackendError> {
        let observed = self.observe()?;
        if observed.iter().all(|item| item.mapped.is_some()) {
            self.spec
                .validate_ready_observed(&observed)
                .map_err(|_| identity_error("system CUDA registered mappings not ready"))?;
        }
        Ok(observed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancellation_is_rechecked_without_claiming_completion() {
        let flag = AtomicBool::new(false);
        assert!(check_cancellation(&flag).is_ok());
        flag.store(true, Ordering::Release);
        let error = check_cancellation(&flag).unwrap_err();
        assert_eq!(error.kind, FailureKind::BackendUnavailable);
        assert_eq!(error.stage, FailureStage::Admission);
    }

    #[test]
    fn cpu_lock_is_not_silently_interpreted_as_system_cuda() {
        let lock = SystemRuntimeLockV6::new(NativeRuntimeSourceV6::Cpu).unwrap();
        assert!(system_spec(&lock).is_err());
    }

    #[test]
    fn malformed_lock_is_rejected_before_any_library_capture() {
        let mut lock = SystemRuntimeLockV6::new(NativeRuntimeSourceV6::Cpu).unwrap();
        lock.canonical_sha256 = "0".repeat(64);
        assert!(system_spec(&lock).is_err());
    }
}
