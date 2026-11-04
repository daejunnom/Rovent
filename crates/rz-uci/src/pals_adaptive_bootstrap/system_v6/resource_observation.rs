//! Bounded, diagnostic V6 startup observations from the original owner.
//!
//! This module does not create a V5 startup owner or V5 receipt. Its caller
//! must pass the exact driver and transferred owner held by the same V6
//! lifecycle. None of these readings proves enforcement, lifetime peaks,
//! numerical acceptance or GPU physical completion.

use super::super::ActualDriver;
use super::driver_assemble::V6TransferredOwners;
use super::driver_prepare::OriginalSnapshotState;
use crate::pals_attestation::checker::resource::{
    self, ProcessResourceSnapshot, ResourceError, ResourceObservation,
};
use crate::search_driver::SearchSessionFailure;
use rz_eval::pals_onnx::PalsAppliedSessionSettings;
use rz_runtime::adaptive::{CpuResourceError, CpuThreadReadback, HostCpuSnapshot};
use rz_search::cpu_checker::ExternalProcessIdentity;
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

/// The source snapshot is retained by the enclosing original lifecycle.
/// A missing/failed snapshot is never replaced by declared session settings.
pub(super) enum FactorySessionsObservationV6 {
    Observed(Arc<[PalsAppliedSessionSettings]>),
    SnapshotPending,
    SnapshotFailed,
    SnapshotDisconnected,
    OwnerMismatch,
    EpochMismatch,
}

/// A skipped probe is distinct from an attempted probe with unknown output.
pub(super) enum ProbeV6<T> {
    NotAttempted(&'static str),
    Attempted(T),
}

/// Errors are preserved as errors rather than substituted with zero values.
pub(super) struct StartupResourceObservationV6 {
    pub(super) process_id: u32,
    pub(super) process_epoch: u64,
    pub(super) original_startup_lock_sha256: String,
    pub(super) started: Instant,
    pub(super) probe_deadline: Instant,
    pub(super) ended: Instant,
    /// Checks the original declaration marker and process epoch. The caller
    /// must still supply the driver held by this exact lifecycle.
    pub(super) owner_scope_consistent: bool,
    pub(super) native_ready_thread:
        ProbeV6<Result<Option<Result<CpuThreadReadback, CpuResourceError>>, SearchSessionFailure>>,
    pub(super) external_identity:
        ProbeV6<Result<Option<ExternalProcessIdentity>, SearchSessionFailure>>,
    pub(super) host_cpu: ProbeV6<Result<HostCpuSnapshot, CpuResourceError>>,
    pub(super) current_process: ProbeV6<Result<Option<ProcessResourceSnapshot>, ResourceError>>,
    pub(super) external_process: ProbeV6<Result<ResourceObservation, ResourceError>>,
    pub(super) factory_sessions: FactorySessionsObservationV6,
}

fn probe<T>(until: Instant, cancelled: &AtomicBool, f: impl FnOnce() -> T) -> ProbeV6<T> {
    if cancelled.load(Ordering::Acquire) {
        ProbeV6::NotAttempted("original_startup_cancelled")
    } else if Instant::now() >= until {
        ProbeV6::NotAttempted("probe_deadline_expired")
    } else {
        ProbeV6::Attempted(f())
    }
}

/// Diagnostic only: call once after the same lifecycle's checker startup
/// and before UCI service. `until` is capped at the original startup
/// deadline and may be shorter. Failures remain in the returned observation
/// and do not reject model execution or replace actual CUDA/GPU checks.
pub(super) fn observe_startup_v6(
    original: &V6TransferredOwners,
    driver: &ActualDriver,
    until: Instant,
) -> StartupResourceObservationV6 {
    let until = until.min(original.cpu.original_until);
    let cancelled = original.cpu.original_cancelled.as_ref();
    let started = Instant::now();
    let process_epoch = driver.process_epoch().0;
    let process_id = std::process::id();
    let original_startup_lock_sha256 = original.cpu.original_source.lock().canonical_sha256.clone();
    let same_owner = Arc::ptr_eq(&original.owner_marker, &original.declaration_marker);
    let owner_scope_consistent = same_owner && original.cpu.clock.epoch().0 == process_epoch;
    let factory_sessions = if !same_owner {
        FactorySessionsObservationV6::OwnerMismatch
    } else if original.cpu.clock.epoch().0 != process_epoch {
        FactorySessionsObservationV6::EpochMismatch
    } else {
        match &original.original_snapshot {
            OriginalSnapshotState::Ready(snapshot)
                if snapshot.process_generation == process_epoch =>
            {
                FactorySessionsObservationV6::Observed(Arc::clone(&snapshot.sessions))
            }
            OriginalSnapshotState::Ready(_) => FactorySessionsObservationV6::EpochMismatch,
            OriginalSnapshotState::Pending => FactorySessionsObservationV6::SnapshotPending,
            OriginalSnapshotState::Failed(_) => FactorySessionsObservationV6::SnapshotFailed,
            OriginalSnapshotState::Disconnected => {
                FactorySessionsObservationV6::SnapshotDisconnected
            }
        }
    };
    let native_ready_thread = if owner_scope_consistent {
        probe(until, cancelled, || {
            driver.native_ready_cpu_thread_observation()
        })
    } else {
        ProbeV6::NotAttempted("owner_scope_inconsistent")
    };
    let external_identity = if owner_scope_consistent {
        probe(until, cancelled, || {
            driver.external_startup_process_identity()
        })
    } else {
        ProbeV6::NotAttempted("owner_scope_inconsistent")
    };
    let host_cpu = probe(until, cancelled, || {
        rz_runtime::adaptive::observe_current_cpu_until(until, cancelled)
    });
    let current_process = probe(until, cancelled, || {
        resource::observe_current_process(until, cancelled)
    });
    let external_process = match &external_identity {
        ProbeV6::Attempted(Ok(Some(identity))) => probe(until, cancelled, || {
            resource::observe(Some(*identity), until, cancelled)
        }),
        ProbeV6::Attempted(Ok(None)) => ProbeV6::NotAttempted("external_identity_unobserved"),
        ProbeV6::Attempted(Err(_)) => ProbeV6::NotAttempted("external_identity_error"),
        ProbeV6::NotAttempted(reason) => ProbeV6::NotAttempted(*reason),
    };
    StartupResourceObservationV6 {
        process_id,
        process_epoch,
        original_startup_lock_sha256,
        started,
        probe_deadline: until,
        ended: Instant::now(),
        owner_scope_consistent,
        native_ready_thread,
        external_identity,
        host_cpu,
        current_process,
        external_process,
        factory_sessions,
    }
}

fn not_attempted(reason: &str) -> serde_json::Value {
    serde_json::json!({"status": "not_attempted", "reason": reason})
}
fn unknown(reason: &str) -> serde_json::Value {
    serde_json::json!({"status": "unknown", "reason": reason})
}
fn observed(value: serde_json::Value) -> serde_json::Value {
    serde_json::json!({"status": "observed", "value": value})
}
fn driver_error(error: &SearchSessionFailure) -> serde_json::Value {
    let physical_completion = match error.physical_completion {
        crate::search_driver::DriverPhysicalCompletion::Confirmed => "confirmed",
        crate::search_driver::DriverPhysicalCompletion::Unknown => "unknown",
    };
    serde_json::json!({"status": "error", "error_type": "search_session_failure", "code": error.code, "detail": error.detail, "reported_physical_completion": physical_completion})
}
fn cpu_error(error: CpuResourceError) -> serde_json::Value {
    use CpuResourceError::*;
    let code = match error {
        InvalidDeclaration => "invalid_declaration",
        SeparateWorkerProfileRequired => "separate_worker_profile_required",
        DedicatedEvidenceRequired => "dedicated_evidence_required",
        InvalidReadback => "invalid_readback",
        AffinityMismatch => "affinity_mismatch",
        ManagerSmtOverlap => "manager_smt_overlap",
        OrtSettingsMismatch => "ort_settings_mismatch",
        ProbeDeadline => "probe_deadline",
        ProbeCanceled => "probe_cancelled",
        ProbeBudgetExceeded => "probe_budget_exceeded",
    };
    serde_json::json!({"status": "error", "error_type": "cpu_resource_error", "code": code})
}
fn resource_error(error: ResourceError) -> serde_json::Value {
    let (code, detail) = match error {
        ResourceError::Unavailable(reason) => ("unavailable", Some(reason)),
        ResourceError::Invalid(reason) => ("invalid", Some(reason)),
        ResourceError::Deadline => ("deadline_or_cancelled", None),
    };
    serde_json::json!({"status":"error", "error_type":"process_resource_error", "code":code, "detail":detail})
}
fn cpu_method(method: rz_runtime::adaptive::CpuProbeMethod) -> &'static str {
    use rz_runtime::adaptive::CpuProbeMethod::*;
    match method {
        LinuxProcThreadStatus => "linux_proc_thread_status",
        LinuxProcProcessStatus => "linux_proc_process_status",
        LinuxSysfsSiblings => "linux_sysfs_siblings",
        LinuxCgroupV2Direct => "linux_cgroup_v2_direct",
        LinuxCgroupV2VisibleAncestors => "linux_cgroup_v2_visible_ancestors",
        WindowsProcessAndThreadGroupAffinity => "windows_process_thread_group_affinity",
        WindowsLogicalProcessorInformationEx => "windows_logical_processor_information_ex",
        WindowsImmediateJobObject => "windows_immediate_job_object",
        WorkerReadyBoundary => "worker_ready_boundary",
        OrtAppliedSessionSettings => "ort_applied_session_settings",
    }
}
fn cpu_unknown(reason: rz_runtime::adaptive::CpuProbeUnknown) -> &'static str {
    use rz_runtime::adaptive::CpuProbeUnknown::*;
    match reason {
        UnsupportedPlatform => "unsupported_platform",
        ReadUnavailable => "read_unavailable",
        InvalidFormat => "invalid_format",
        BudgetExceeded => "budget_exceeded",
        NamespaceOrMountAmbiguous => "namespace_or_mount_ambiguous",
        ChangedDuringProbe => "changed_during_probe",
        HiddenAncestors => "hidden_ancestors",
        WindowsJobHierarchyUnavailable => "windows_job_hierarchy_unavailable",
        WindowsGroupAffinityAmbiguous => "windows_group_affinity_ambiguous",
        UnobservedThread => "unobserved_thread",
        UnobservedOrtSettings => "unobserved_ort_settings",
    }
}
fn cpu_readback<T>(
    value: &rz_runtime::adaptive::CpuReadback<T>,
    render: impl FnOnce(&T) -> serde_json::Value,
) -> serde_json::Value {
    match value {
        rz_runtime::adaptive::CpuReadback::Known { value, method } => {
            serde_json::json!({"status":"observed", "method":cpu_method(*method), "value":render(value)})
        }
        rz_runtime::adaptive::CpuReadback::Unknown(reason) => unknown(cpu_unknown(*reason)),
    }
}
fn cpus(value: &[rz_runtime::adaptive::LogicalCpu]) -> serde_json::Value {
    serde_json::Value::Array(
        value
            .iter()
            .map(|cpu| serde_json::json!({"group":cpu.group,"index":cpu.index}))
            .collect(),
    )
}
fn quota(value: &rz_runtime::adaptive::CpuQuotaReadback) -> serde_json::Value {
    use rz_runtime::adaptive::CpuQuotaReadback::*;
    match value {
        Unlimited => serde_json::json!({"kind":"unlimited"}),
        Limited {
            quota_us,
            period_us,
        } => serde_json::json!({"kind":"limited", "quota_us":quota_us, "period_us":period_us}),
        WindowsHardCap { rate_per_10000 } => {
            serde_json::json!({"kind":"windows_hard_cap", "rate_per_10000":rate_per_10000})
        }
    }
}
fn host(value: &HostCpuSnapshot) -> serde_json::Value {
    use rz_runtime::adaptive::CpuAffinityScope;
    let scope = match value.process_affinity_scope {
        CpuAffinityScope::Process => "process",
        CpuAffinityScope::ThreadGroupLeader => "thread_group_leader",
        CpuAffinityScope::Unknown => "unknown",
    };
    serde_json::json!({
        "process_id":value.process_id, "caller_thread_id":cpu_readback(&value.caller_thread_id, |id| serde_json::json!(id)),
        "process_affinity":cpu_readback(&value.process_affinity, |ids| cpus(ids)), "process_affinity_scope":scope,
        "caller_affinity":cpu_readback(&value.caller_affinity, |ids| cpus(ids)),
        "physical_cores":cpu_readback(&value.physical_cores, |cores| serde_json::Value::Array(cores.iter().map(|core| cpus(&core.logical_cpus)).collect())),
        "direct_cpu_quota":cpu_readback(&value.direct_cpu_quota, quota),
        "visible_ancestor_cpu_quota":cpu_readback(&value.visible_ancestor_cpu_quota, quota),
        "ancestor_scope":cpu_unknown(value.ancestor_scope), "sequential_snapshot":value.sequential_snapshot
    })
}
fn thread(value: &CpuThreadReadback) -> serde_json::Value {
    use rz_runtime::adaptive::CpuActor;
    let actor = match value.actor {
        CpuActor::NativeCaller => serde_json::json!({"kind":"native_caller"}),
        CpuActor::Manager => serde_json::json!({"kind":"manager"}),
        CpuActor::OwnWorker(index) => serde_json::json!({"kind":"own_worker", "index":index}),
        CpuActor::ExternalEngine(index) => {
            serde_json::json!({"kind":"external_engine", "index":index})
        }
        CpuActor::RuntimeHelper(index) => {
            serde_json::json!({"kind":"runtime_helper", "index":index})
        }
    };
    serde_json::json!({"actor":actor, "thread_id":value.thread_id, "affinity":cpu_readback(&value.affinity, |ids| cpus(ids))})
}
fn sessions(value: &FactorySessionsObservationV6) -> serde_json::Value {
    match value {
        FactorySessionsObservationV6::Observed(sessions) => observed(serde_json::Value::Array(sessions.iter().map(|session| {
            let provider = match session.provider() {
                rz_eval::onnx::Provider::Cpu => serde_json::json!({"kind":"cpu"}),
                rz_eval::onnx::Provider::Cuda { device_id, arena_bytes } => serde_json::json!({"kind":"cuda", "device_id":device_id, "arena_bytes":arena_bytes}),
            };
            serde_json::json!({"role":session.role(), "provider":provider, "intra_threads":session.intra_threads(), "inter_threads":session.inter_threads(), "intra_op_spinning":session.intra_op_spinning(), "inter_op_spinning":session.inter_op_spinning(), "parallel_execution":session.parallel_execution(), "cuda_provider_available":session.cuda_provider_available(), "evidence_scope":session.evidence_scope(), "coverage_scope":session.coverage_scope(), "evidence_method":session.evidence_method()})
        }).collect())),
        FactorySessionsObservationV6::SnapshotPending => unknown("factory_snapshot_pending"),
        FactorySessionsObservationV6::SnapshotFailed => unknown("factory_snapshot_failed"),
        FactorySessionsObservationV6::SnapshotDisconnected => unknown("factory_snapshot_disconnected"),
        FactorySessionsObservationV6::OwnerMismatch => unknown("owner_mismatch"),
        FactorySessionsObservationV6::EpochMismatch => unknown("epoch_mismatch"),
    }
}

impl StartupResourceObservationV6 {
    /// Standalone diagnostic value. The caller owns publication and must not
    /// use this object as a V5 receipt or as evidence of placement, numerical
    /// correctness, a physical fence, or measured performance.
    pub(super) fn to_receipt_json(&self) -> serde_json::Value {
        let native = match &self.native_ready_thread {
            ProbeV6::NotAttempted(reason) => not_attempted(reason),
            ProbeV6::Attempted(Err(error)) => driver_error(error),
            ProbeV6::Attempted(Ok(None)) => unknown("native_ready_thread_not_observed"),
            ProbeV6::Attempted(Ok(Some(Err(error)))) => cpu_error(*error),
            ProbeV6::Attempted(Ok(Some(Ok(value)))) => observed(thread(value)),
        };
        let external_identity = match &self.external_identity {
            ProbeV6::NotAttempted(reason) => not_attempted(reason),
            ProbeV6::Attempted(Err(error)) => driver_error(error),
            ProbeV6::Attempted(Ok(None)) => unknown("external_identity_not_observed"),
            ProbeV6::Attempted(Ok(Some(identity))) => observed(serde_json::json!({
                "pid":identity.pid, "process_group":identity.process_group,
                "proc_start_ticks":identity.proc_start_ticks,
            })),
        };
        let host_cpu = match &self.host_cpu {
            ProbeV6::NotAttempted(reason) => not_attempted(reason),
            ProbeV6::Attempted(Err(error)) => cpu_error(*error),
            ProbeV6::Attempted(Ok(value)) => observed(host(value)),
        };
        let current_process = match &self.current_process {
            ProbeV6::NotAttempted(reason) => not_attempted(reason),
            ProbeV6::Attempted(Err(error)) => resource_error(*error),
            ProbeV6::Attempted(Ok(None)) => unknown("current_process_unavailable"),
            ProbeV6::Attempted(Ok(Some(value))) => observed(serde_json::json!(value)),
        };
        let external_process = match &self.external_process {
            ProbeV6::NotAttempted(reason) => not_attempted(reason),
            ProbeV6::Attempted(Err(error)) => resource_error(*error),
            ProbeV6::Attempted(Ok(value)) => match &value.snapshot {
                Some(snapshot) => observed(serde_json::json!(snapshot)),
                None => unknown(value.unavailable.unwrap_or("external_resource_unavailable")),
            },
        };
        let elapsed_us = self
            .ended
            .saturating_duration_since(self.started)
            .as_micros();
        let elapsed_us = u64::try_from(elapsed_us).ok();
        let requested_probe_budget_us = u64::try_from(
            self.probe_deadline
                .saturating_duration_since(self.started)
                .as_micros(),
        )
        .ok();
        serde_json::json!({
            "domain":"rz-pals-system-v6-startup-resource-observation/1",
            "scope":"same_lifecycle_ready_boundary_diagnostic_not_lifetime",
            "process_id":self.process_id,
            "process_epoch":self.process_epoch,
            "original_startup_lock_sha256":self.original_startup_lock_sha256,
            "lock_identity_scope":"original_pinned_owner_identity_not_source_reproduction_or_physical_fence",
            "owner_scope_consistency":self.owner_scope_consistent,
            "owner_scope_note":"declaration marker and process epoch only; caller supplies the same original lifecycle driver",
            "observation_elapsed_us":elapsed_us,
            "requested_probe_budget_us": requested_probe_budget_us,
            "probe_budget_scope":"cooperative_deadline_not_kernel_io_preemption",
            "native_ready_thread":native,
            "external_identity":external_identity,
            "host_cpu":host_cpu,
            "current_process":current_process,
            "external_process":external_process,
            "factory_session_settings":sessions(&self.factory_sessions),
            "vram_peak_bytes":unknown("not_observed"),
            "placement_acceptance":"not_evaluated",
            "numerical_acceptance":"not_evaluated",
            "physical_completion":"unknown",
            "performance_acceptance":"not_evaluated",
            "source_scope":"original_v6_factory_snapshot_and_same_driver_ready_observations",
        })
    }
}
