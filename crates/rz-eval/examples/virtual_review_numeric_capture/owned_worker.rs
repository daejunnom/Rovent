//! System CUDA capture on a single retained startup and execution owner.
//!
//! This module is intentionally not a process supervisor or device-wide
//! fence. A quarantined worker retains its original custody only while the
//! containing process remains alive.

use super::{Command, LoadedPolicyObservation, Outcome};
use rz_eval::error::{BackendError, CauseCode, FailureKind, FailureStage};
use rz_eval::onnx::{OrtRuntime, SystemRuntime};
use rz_eval::pals_onnx::virtual_review::{
    PalsVirtualReviewOnnxBackend, VirtualReviewArtifactKind, VirtualReviewExecutionMode,
    VirtualReviewStartupCustody,
};
use rz_eval::pals_onnx::{PalsCudaControlPolicy, PalsOnnxConfig};
use rz_eval::worker::{PhysicalLease, PhysicalRecoveryReservation, PhysicalRun, SingleWorker};
use std::time::Instant;

pub(super) enum CaptureJob {
    Start,
    Native(Command),
}

pub(super) struct Started {
    pub epoch: [u8; 32],
    pub backend_digest: [u8; 32],
    pub policy_observation: LoadedPolicyObservation,
}

pub(super) enum CaptureReply {
    Started(Started),
    Native(Outcome),
}

pub(super) type Worker = SingleWorker<CaptureJob, Result<CaptureReply, BackendError>>;
pub(super) type Lease = PhysicalLease<CaptureJob, Result<CaptureReply, BackendError>>;

/// The caller owns the returned lease and must retain it on pending or unknown
/// completion; no logical timeout is treated as permission to discard it.
pub(super) fn submit_start(worker: &mut Worker) -> Result<Lease, BackendError> {
    worker.submit(CaptureJob::Start)
}

pub(super) fn submit_native(worker: &mut Worker, command: Command) -> Result<Lease, BackendError> {
    worker.submit(CaptureJob::Native(command))
}

/// All fields describe the actual startup to run on the same retained owner.
/// The caller must separately retain its original process/device supervision.
pub(super) struct Startup {
    pub runtime: SystemRuntime,
    pub expected_runtime_sha256: [u8; 32],
    pub expected_manifest_sha256: String,
    pub config: PalsOnnxConfig,
    pub mode: VirtualReviewExecutionMode,
    pub kind: VirtualReviewArtifactKind,
    pub policy: Option<PalsCudaControlPolicy>,
    pub requested_cudnn_conv_search: Option<String>,
    pub until: Instant,
}

fn error(kind: FailureKind, detail: &'static str) -> BackendError {
    BackendError::new(kind, FailureStage::Admission, detail)
}

/// On thread creation failure `original` remains in the caller's slot and the
/// reservation remains with that caller. Once started, the worker registry
/// owns the exact custody and the shutdown proof is obtained at shutdown time.
pub(super) fn spawn(
    original: &mut Option<VirtualReviewStartupCustody>,
    reservation: &mut PhysicalRecoveryReservation,
    mut startup: Startup,
) -> Result<Worker, BackendError> {
    // These pure checks precede worker creation and every ORT call. Their
    // failure leaves the caller's original owner and reservation in place.
    if Instant::now() >= startup.until {
        return Err(error(
            FailureKind::BackendUnavailable,
            "system capture expired before worker creation",
        ));
    }
    startup.config.validate_system_execution_selection()?;
    rz_eval::asset::parse_sha256(&startup.expected_manifest_sha256)?;
    let libraries = startup.runtime.libraries();
    if !libraries.iter().any(|item| {
        item.role == "ort-core"
            && item.required_mapping
            && item.sha256 == startup.expected_runtime_sha256
    }) || !libraries
        .iter()
        .any(|item| item.role == "ort-provider-cuda" && item.required_mapping)
    {
        return Err(error(
            FailureKind::IdentityMismatch,
            "system capture ORT core or CUDA provider differs",
        ));
    }
    let mut attempted = false;
    // An install failure returns the real runtime; it must remain with this
    // same worker closure, rather than be dropped or reconstructed.
    let mut rejected_runtime: Option<OrtRuntime> = None;
    SingleWorker::spawn_with_retained_synchronous_owner(
        original,
        reservation,
        move |custody, job| {
            if rejected_runtime.is_some() {
                return PhysicalRun::Quarantined(error(
                    FailureKind::BackendFailure,
                    "original rejected runtime remains in worker custody",
                ));
            }
            match job {
                CaptureJob::Start => {
                    if attempted {
                        return PhysicalRun::Complete(Err(error(
                            FailureKind::InvalidInput,
                            "system capture startup was already attempted",
                        )));
                    }
                    attempted = true;
                    if Instant::now() >= startup.until {
                        return PhysicalRun::Complete(Err(error(
                            FailureKind::BackendUnavailable,
                            "system capture expired before native startup",
                        )));
                    }
                    // This is the first ORT native boundary. `startup.runtime`
                    // remains held by the worker closure on error or unwind.
                    let runtime = match OrtRuntime::load_system_cuda_with_checked_entry(
                        &startup.runtime,
                        || {
                            if Instant::now() >= startup.until {
                                Err(error(
                                    FailureKind::BackendUnavailable,
                                    "system capture expired immediately before ORT entry",
                                ))
                            } else {
                                Ok(())
                            }
                        },
                    ) {
                        Ok(runtime) => runtime,
                        Err(cause) => return PhysicalRun::Quarantined(cause),
                    };
                    let actual_runtime_sha256 = runtime.binary_digest();
                    if let Err((cause, returned)) = custody.install_runtime(runtime) {
                        rejected_runtime = Some(returned);
                        return PhysicalRun::Quarantined(cause);
                    }
                    if actual_runtime_sha256 != startup.expected_runtime_sha256 {
                        return PhysicalRun::Quarantined(error(
                            FailureKind::IdentityMismatch,
                            "original system capture ORT identity differs",
                        ));
                    }
                    let explicit_control = startup.policy.is_some();
                    let policy = startup.policy.take();
                    let result =
                        PalsVirtualReviewOnnxBackend::load_owned_sources_with_profile_into_checked(
                            custody,
                            &startup.expected_manifest_sha256,
                            startup.config,
                            startup.mode,
                            startup.kind,
                            policy,
                            &mut || {
                                if Instant::now() >= startup.until {
                                    Err(error(
                                        FailureKind::BackendUnavailable,
                                        "system capture expired before a session constructor",
                                    ))
                                } else {
                                    Ok(())
                                }
                            },
                        );
                    if let Err(cause) = result {
                        return PhysicalRun::Quarantined(cause);
                    }
                    let Some(backend) = custody.backend() else {
                        return PhysicalRun::Quarantined(error(
                            FailureKind::BackendFailure,
                            "original system capture backend missing after startup",
                        ));
                    };
                    let policy_observation = match super::loaded_policy_observation(
                        backend,
                        startup.requested_cudnn_conv_search.as_deref(),
                        explicit_control,
                    ) {
                        Ok(value) => value,
                        Err(cause) => {
                            return PhysicalRun::Complete(Err(error(
                                FailureKind::BackendFailure,
                                "cannot observe loaded system policy",
                            )
                            .with_external_cause(CauseCode::SessionConfiguration, &cause)));
                        }
                    };
                    PhysicalRun::Complete(Ok(CaptureReply::Started(Started {
                        epoch: backend.model_epoch(),
                        backend_digest: backend.backend_digest(),
                        policy_observation,
                    })))
                }
                CaptureJob::Native(command) => {
                    let Some(backend) = custody.backend_slot_mut().as_mut() else {
                        return PhysicalRun::Complete(Err(error(
                            FailureKind::InvalidInput,
                            "system capture backend has not started",
                        )));
                    };
                    match backend.execute_owned_command(command) {
                        PhysicalRun::Complete(result) => {
                            PhysicalRun::Complete(result.map(CaptureReply::Native))
                        }
                        PhysicalRun::Quarantined(cause) => PhysicalRun::Quarantined(cause),
                    }
                }
            }
        },
        // The proof must be requested from the same owner *now*, never cached
        // before the final run. It is only a normal same-session shutdown.
        |custody, _until| custody.synchronous_shutdown_proof(),
    )
}
