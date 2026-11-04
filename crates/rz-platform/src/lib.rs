//! Small OS CPU observations and explicit current-thread affinity ownership.
//!
//! Native calls stay in the platform boundary. Existing observation functions
//! remain read-only. Only an explicitly requested current-thread guard changes
//! affinity; it grants no process-tree quota, placement or startup authority.
#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(target_os = "windows")]
mod windows;
#[cfg(target_os = "windows")]
pub use windows::{observe_windows_cpu, observe_windows_thread_affinity};

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "linux")]
mod managed_output;
#[cfg(target_os = "linux")]
pub use linux::{
    duplicate_inherited_unix_datagram, mark_inherited_datagram_cloexec,
    observe_linux_thread_affinity, seal_exclusive_inherited_unix_datagram,
};
#[cfg(target_os = "linux")]
pub use managed_output::{
    managed_output_session_nonce, ManagedCollectionInheritedFds, ManagedInputInheritedFds,
    ManagedOutputInheritedFds, ManagedOutputPacket, ManagedOutputPeerCredentials,
};

#[cfg(any(target_os = "linux", target_os = "windows"))]
mod placement;
#[cfg(any(target_os = "linux", target_os = "windows"))]
pub use placement::{
    apply_current_thread_affinity, CurrentThreadAffinityGuard, ThreadAffinityApplyError,
    ThreadAffinityCause, ThreadAffinityCompensationOutcome, ThreadAffinityFailure,
    ThreadAffinityFailureRecord, ThreadAffinityMask, ThreadAffinityRestoreError,
    ThreadAffinityRollback, ThreadAffinitySetConflict, ThreadAffinitySnapshot, ThreadAffinityStage,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LinuxThreadAffinity {
    pub process_id: u32,
    pub thread_id: u64,
    pub cpu_ids: Vec<u16>,
    pub identity_stable: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlatformCpuError {
    NativeCall {
        function: &'static str,
        os_code: u32,
    },
    InvalidNativeBuffer,
    NativeBufferBudget,
    AmbiguousProcessGroups,
    UnsupportedArchitecture,
    UnsupportedJobRatePolicy,
    InvalidText,
    ReadUnavailable,
    ChangedDuringProbe,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WindowsDirectCpuQuota {
    Unlimited,
    HardCap { rate_per_10000: u16 },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WindowsCpuSnapshot {
    pub process_id: u32,
    pub caller_thread_id: u64,
    pub identity_stable: bool,
    pub process_group_masks: Result<Vec<(u16, u64)>, PlatformCpuError>,
    pub caller_group_mask: Result<(u16, u64), PlatformCpuError>,
    pub physical_core_group_masks: Result<Vec<Vec<(u16, u64)>>, PlatformCpuError>,
    pub immediate_job_cpu_quota: Result<WindowsDirectCpuQuota, PlatformCpuError>,
    /// QueryInformationJobObject(NULL) reads only the immediate Job Object.
    pub effective_job_hierarchy_known: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WindowsThreadAffinity {
    pub process_id: u32,
    pub thread_id: u64,
    pub group_mask: (u16, u64),
    pub identity_stable: bool,
}
