//! Resource declaration and bounded readback for one NoGame smoke owner.
//!
//! This code installs no cgroup, affinity, swap, CPU or memory settings. A
//! caller may pass its original cgroup2 directory FD for observation. A
//! snapshot is neither a pre-exec placement barrier nor physical completion.

use super::{NoGameExecutionStartV1, ProxyEntrySmokeSnapshotOwnerV1};
use crate::{ArenaError, process::SpawnCustody};
use cap_fs_ext::{FollowSymlinks, OpenOptionsFollowExt};
use cap_std::fs::{Dir, OpenOptions};
use nix::{
    sys::{
        statfs::{CGROUP2_SUPER_MAGIC, fstatfs},
        wait::{Id, WaitPidFlag, WaitStatus, waitid},
    },
    unistd::Pid,
};
use std::{fs::File, io::Read, sync::Arc};

/// Separate explicit declaration, in millicpus and bytes. No swap policy or
/// fixed CPU numbering is inferred. Registration does not apply these values.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct NoGameResourceLimitsV1 {
    pub cpu_quota_millicpus: u32,
    pub memory_high_bytes: u64,
    pub memory_max_bytes: u64,
}

impl NoGameResourceLimitsV1 {
    fn valid(&self) -> bool {
        self.cpu_quota_millicpus > 0
            && self.memory_high_bytes > 0
            && self.memory_high_bytes <= self.memory_max_bytes
    }
}

/// Readback values distinguish a kernel `max` from an unavailable value.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ResourceLimitReadbackV1 {
    Numeric(u64),
    Unlimited,
    Unknown,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ResourceComparisonV1 {
    Match,
    Mismatch,
    Unknown,
}

#[derive(Clone, Debug)]
pub(crate) struct NoGameResourceObservationV1 {
    pub declared: NoGameResourceLimitsV1,
    pub cgroup2_fd_observed: bool,
    pub cpu_quota_us: ResourceLimitReadbackV1,
    pub cpu_period_us: Option<u64>,
    pub memory_high_bytes: ResourceLimitReadbackV1,
    pub memory_max_bytes: ResourceLimitReadbackV1,
    pub cpu_comparison: ResourceComparisonV1,
    pub high_comparison: ResourceComparisonV1,
    pub max_comparison: ResourceComparisonV1,
    /// Membership in this exact cgroup, not ancestry or continued placement.
    pub parent_direct_member: Option<bool>,
    pub child_direct_member: Option<bool>,
    /// Current caller thread only. This does not describe all process threads.
    pub caller_thread_affinity: Option<Vec<u16>>,
    /// This owner only reads; a successful read does not mean it installed it.
    pub installed_by_this_owner: bool,
    pub scope: &'static str,
}

/// A rejected bind returns the exact original optional cgroup FD to its owner.
#[must_use = "retain the original cgroup descriptor on binding failure"]
pub(crate) struct NoGameResourceBindFailure {
    pub error: ArenaError,
    pub cgroup: Option<File>,
}

/// Same original snapshot/clock and optional original cgroup FD. A scope may
/// also be shared by an external supervisor; exclusivity is not asserted.
#[must_use = "retain with the same original smoke owner until settled"]
pub(crate) struct NoGameResourceOwnerV1 {
    snapshot: Arc<ProxyEntrySmokeSnapshotOwnerV1>,
    execution: Arc<NoGameExecutionStartV1>,
    limits: NoGameResourceLimitsV1,
    cgroup: Option<File>,
}

impl NoGameResourceOwnerV1 {
    pub(crate) fn bind_original(
        snapshot: &Arc<ProxyEntrySmokeSnapshotOwnerV1>,
        execution: &Arc<NoGameExecutionStartV1>,
        limits: NoGameResourceLimitsV1,
        cgroup: Option<File>,
    ) -> Result<Self, NoGameResourceBindFailure> {
        if !limits.valid()
            || !Arc::ptr_eq(&snapshot.execution, execution)
            || !execution.matches_plan(&snapshot.plan)
        {
            return Err(NoGameResourceBindFailure {
                error: ArenaError::Invalid("NoGame resource declaration or owner differs".into()),
                cgroup,
            });
        }
        Ok(Self {
            snapshot: Arc::clone(snapshot),
            execution: Arc::clone(execution),
            limits,
            cgroup,
        })
    }

    pub(crate) fn matches_original(
        &self,
        snapshot: &Arc<ProxyEntrySmokeSnapshotOwnerV1>,
        execution: &Arc<NoGameExecutionStartV1>,
    ) -> bool {
        Arc::ptr_eq(&self.snapshot, snapshot)
            && Arc::ptr_eq(&self.execution, execution)
            && execution.matches_plan(&snapshot.plan)
    }

    /// An unavailable read or non-live child remains Unknown and never
    /// prevents an independent CPU task or changes any kernel control.
    pub(crate) fn observe(&self, child: Option<&SpawnCustody>) -> NoGameResourceObservationV1 {
        let mut result = NoGameResourceObservationV1 {
            declared: self.limits.clone(),
            cgroup2_fd_observed: false,
            cpu_quota_us: ResourceLimitReadbackV1::Unknown,
            cpu_period_us: None,
            memory_high_bytes: ResourceLimitReadbackV1::Unknown,
            memory_max_bytes: ResourceLimitReadbackV1::Unknown,
            cpu_comparison: ResourceComparisonV1::Unknown,
            high_comparison: ResourceComparisonV1::Unknown,
            max_comparison: ResourceComparisonV1::Unknown,
            parent_direct_member: None,
            child_direct_member: None,
            caller_thread_affinity: rz_platform::observe_linux_thread_affinity()
                .ok()
                .map(|affinity| affinity.cpu_ids),
            installed_by_this_owner: false,
            scope: "bounded_nonatomic_original_cgroup_fd_and_caller_thread_snapshot_not_enforcement_or_lifetime_proof",
        };
        let Some(file) = self.cgroup.as_ref() else {
            return result;
        };
        let Ok(metadata) = file.metadata() else {
            return result;
        };
        if !metadata.is_dir()
            || !fstatfs(file).is_ok_and(|stat| stat.filesystem_type() == CGROUP2_SUPER_MAGIC)
        {
            return result;
        }
        let Ok(original_fd) = file.try_clone() else {
            return result;
        };
        let directory = Dir::from_std_file(original_fd);
        result.cgroup2_fd_observed = true;
        let (quota, period) = parse_cpu(read_bounded(&directory, "cpu.max", 128).as_deref());
        result.cpu_quota_us = quota;
        result.cpu_period_us = period;
        result.memory_high_bytes =
            parse_limit(read_bounded(&directory, "memory.high", 128).as_deref());
        result.memory_max_bytes =
            parse_limit(read_bounded(&directory, "memory.max", 128).as_deref());
        result.cpu_comparison = compare_cpu(
            &result.cpu_quota_us,
            period,
            self.limits.cpu_quota_millicpus,
        );
        result.high_comparison =
            compare_limit(&result.memory_high_bytes, self.limits.memory_high_bytes);
        result.max_comparison =
            compare_limit(&result.memory_max_bytes, self.limits.memory_max_bytes);
        let child_pid = child
            .and_then(|owner| owner.child())
            .map(|process| process.id());
        let live_before = child_pid.is_some_and(is_owned_live_child);
        if let Some(procs) = read_bounded(&directory, "cgroup.procs", 64 * 1024)
            && let Some(pids) = parse_pids(&procs)
        {
            result.parent_direct_member = Some(pids.contains(&std::process::id()));
            if let Some(pid) = child_pid
                && live_before
                && is_owned_live_child(pid)
            {
                result.child_direct_member = Some(pids.contains(&pid));
            }
        }
        result
    }
}

fn read_bounded(directory: &Dir, name: &str, cap: usize) -> Option<String> {
    let mut options = OpenOptions::new();
    options.read(true).follow(FollowSymlinks::No);
    let file = directory.open_with(name, &options).ok()?;
    let mut data = Vec::new();
    file.take(u64::try_from(cap).ok()?.checked_add(1)?)
        .read_to_end(&mut data)
        .ok()?;
    if data.len() > cap {
        return None;
    }
    String::from_utf8(data).ok()
}

fn parse_limit(value: Option<&str>) -> ResourceLimitReadbackV1 {
    match value.map(str::trim) {
        Some("max") => ResourceLimitReadbackV1::Unlimited,
        Some(value) if !value.is_empty() && value.bytes().all(|b| b.is_ascii_digit()) => value
            .parse()
            .map(ResourceLimitReadbackV1::Numeric)
            .unwrap_or(ResourceLimitReadbackV1::Unknown),
        _ => ResourceLimitReadbackV1::Unknown,
    }
}

fn parse_cpu(value: Option<&str>) -> (ResourceLimitReadbackV1, Option<u64>) {
    let Some(value) = value else {
        return (ResourceLimitReadbackV1::Unknown, None);
    };
    let mut fields = value.split_ascii_whitespace();
    let quota = parse_limit(fields.next());
    let period = fields
        .next()
        .filter(|value| value.bytes().all(|byte| byte.is_ascii_digit()))
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|period| *period > 0);
    if fields.next().is_some() || period.is_none() {
        return (ResourceLimitReadbackV1::Unknown, None);
    }
    (quota, period)
}

fn compare_limit(value: &ResourceLimitReadbackV1, requested: u64) -> ResourceComparisonV1 {
    match value {
        ResourceLimitReadbackV1::Numeric(actual) if *actual == requested => {
            ResourceComparisonV1::Match
        }
        ResourceLimitReadbackV1::Numeric(_) | ResourceLimitReadbackV1::Unlimited => {
            ResourceComparisonV1::Mismatch
        }
        ResourceLimitReadbackV1::Unknown => ResourceComparisonV1::Unknown,
    }
}

fn compare_cpu(
    quota: &ResourceLimitReadbackV1,
    period: Option<u64>,
    requested_millicpus: u32,
) -> ResourceComparisonV1 {
    match (quota, period) {
        (ResourceLimitReadbackV1::Numeric(quota), Some(period)) => {
            if u128::from(*quota) * 1000 == u128::from(period) * u128::from(requested_millicpus) {
                ResourceComparisonV1::Match
            } else {
                ResourceComparisonV1::Mismatch
            }
        }
        (ResourceLimitReadbackV1::Unlimited, Some(_)) => ResourceComparisonV1::Mismatch,
        _ => ResourceComparisonV1::Unknown,
    }
}

fn parse_pids(raw: &str) -> Option<Vec<u32>> {
    let mut pids = Vec::new();
    for line in raw.lines() {
        let pid = line.parse::<u32>().ok()?;
        if pid == 0 {
            return None;
        }
        pids.push(pid);
    }
    Some(pids)
}

fn is_owned_live_child(pid: u32) -> bool {
    let Ok(pid) = i32::try_from(pid) else {
        return false;
    };
    matches!(
        waitid(
            Id::Pid(Pid::from_raw(pid)),
            WaitPidFlag::WEXITED | WaitPidFlag::WNOHANG | WaitPidFlag::WNOWAIT,
        ),
        Ok(WaitStatus::StillAlive)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn declared_limits_and_kernel_readback_remain_distinct() {
        let policy = NoGameResourceLimitsV1 {
            cpu_quota_millicpus: 2000,
            memory_high_bytes: 6_u64 << 30,
            memory_max_bytes: 12_u64 << 30,
        };
        assert!(policy.valid());
        let (quota, period) = parse_cpu(Some("200000 100000\n"));
        assert_eq!(
            compare_cpu(&quota, period, policy.cpu_quota_millicpus),
            ResourceComparisonV1::Match
        );
        assert_eq!(
            compare_limit(&ResourceLimitReadbackV1::Unlimited, policy.memory_max_bytes),
            ResourceComparisonV1::Mismatch
        );
        assert_eq!(
            compare_limit(&parse_limit(None), policy.memory_high_bytes),
            ResourceComparisonV1::Unknown
        );
        assert_eq!(
            compare_cpu(
                &parse_cpu(Some("max 100000")).0,
                Some(100000),
                policy.cpu_quota_millicpus
            ),
            ResourceComparisonV1::Mismatch
        );
    }

    #[test]
    fn malformed_or_unavailable_kernel_values_are_unknown() {
        assert_eq!(
            parse_cpu(Some("100 0")),
            (ResourceLimitReadbackV1::Unknown, None)
        );
        assert_eq!(
            parse_cpu(Some("100 200 extra")),
            (ResourceLimitReadbackV1::Unknown, None)
        );
        assert_eq!(
            parse_limit(Some("unavailable")),
            ResourceLimitReadbackV1::Unknown
        );
        assert!(parse_pids("12\ninvalid\n").is_none());
    }
}
