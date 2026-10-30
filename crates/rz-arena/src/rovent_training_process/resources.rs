//! Linux delegated-cgroup owner for an explicitly launched Rovent trainer.
//!
//! Only a fresh child leaf is configured. The caller retains this owner (also
//! on failure); neither Drop nor observation moves, kills or removes anything.
//! Membership/affinity/procs readings are bounded snapshots, not lifetime proof.
#![cfg(target_os = "linux")]

use crate::ArenaError;
use cap_fs_ext::{DirExt, FollowSymlinks, OpenOptionsFollowExt};
use cap_std::{ambient_authority, fs::Dir};
use nix::sched::{CpuSet, sched_getaffinity};
use nix::sys::stat::{Mode, mkdirat};
use nix::sys::statfs::{CGROUP2_SUPER_MAGIC, PROC_SUPER_MAGIC, fstatfs};
use nix::unistd::{Pid, Uid};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;
use std::fs::File;
use std::io::{Read, Write};
use std::os::unix::fs::MetadataExt;
use std::path::{Component, Path, PathBuf};

const MAX_PATH: usize = 4096;
const MAX_PROC_TEXT: usize = 16 * 1024;
const MAX_MOUNTINFO: usize = 256 * 1024;
const MAX_CGROUP_PROCS: usize = 64 * 1024;
const CPU_PERIOD_US: u64 = 1_000_000;

mod native_tree;
pub use native_tree::{
    RoventNativeProcessEntry, RoventNativeProcessSnapshot, RoventNativeTreeObservation,
};

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventTrainingResourcePolicy {
    pub cpu_quota_millicpus: u32,
    pub memory_high_bytes: u64,
    pub memory_max_bytes: u64,
    pub pids_max: u32,
    pub affinity: Vec<u32>,
    pub device: String,
}
impl RoventTrainingResourcePolicy {
    pub fn from_json(input: &str) -> Result<Self, ArenaError> {
        let value: Self = crate::decode_json(input)?;
        value.validate()?;
        Ok(value)
    }
    pub fn validate(&self) -> Result<(), ArenaError> {
        let cpus: BTreeSet<_> = self.affinity.iter().copied().collect();
        let device = self.device == "cpu"
            || self
                .device
                .strip_prefix("cuda:")
                .is_some_and(|n| canonical_u64(n).is_ok_and(|v| v < 64));
        check(
            (1..=256_000).contains(&self.cpu_quota_millicpus)
                && self.memory_high_bytes > 0
                && self.memory_high_bytes <= self.memory_max_bytes
                && self.memory_max_bytes <= 1_u64 << 50
                && (1..=4096).contains(&self.pids_max)
                && !self.affinity.is_empty()
                && self.affinity.len() <= 1024
                && cpus.len() == self.affinity.len()
                && self
                    .affinity
                    .iter()
                    .all(|&c| c < 1024 && CpuSet::new().set(c as usize).is_ok())
                && device,
            "invalid bounded CPU/memory/pids/affinity/device policy",
        )
    }
    fn cpu_quota_us(&self) -> u64 {
        // A 1s period represents even one millicpu with Linux's minimum 1ms
        // quota. The actual period is published; it is not silently rounded.
        u64::from(self.cpu_quota_millicpus) * (CPU_PERIOD_US / 1000)
    }
}

#[derive(Clone, Copy, Debug, Serialize, Eq, PartialEq)]
struct DirectoryIdentity {
    device: u64,
    inode: u64,
}
impl DirectoryIdentity {
    fn of(file: &File) -> Result<Self, ArenaError> {
        let m = file.metadata().map_err(|_| io("directory_metadata"))?;
        check(m.is_dir(), "resource descriptor is not a directory")?;
        Ok(Self {
            device: m.dev(),
            inode: m.ino(),
        })
    }
}

#[derive(Clone, Debug, Serialize, Eq, PartialEq)]
pub struct RoventTrainingResourceReadback {
    pub cpu_quota_us: u64,
    pub cpu_period_us: u64,
    pub memory_high_bytes: u64,
    pub memory_max_bytes: u64,
    pub memory_swap_max_bytes: u64,
    pub pids_max: u64,
}
impl RoventTrainingResourceReadback {
    fn read(directory: &Dir) -> Result<Self, ArenaError> {
        let cpu = read_kernel(directory, "cpu.max", 128, false)?;
        let mut fields = cpu.split_whitespace();
        let quota = fields
            .next()
            .ok_or_else(|| invalid("cpu.max quota absent"))?;
        let period = fields
            .next()
            .ok_or_else(|| invalid("cpu.max period absent"))?;
        check(fields.next().is_none(), "cpu.max field count")?;
        Ok(Self {
            cpu_quota_us: canonical_u64(quota)?,
            cpu_period_us: canonical_u64(period)?,
            memory_high_bytes: read_number(directory, "memory.high")?,
            memory_max_bytes: read_number(directory, "memory.max")?,
            memory_swap_max_bytes: read_number(directory, "memory.swap.max")?,
            pids_max: read_number(directory, "pids.max")?,
        })
    }
    fn matches(&self, policy: &RoventTrainingResourcePolicy) -> bool {
        self.cpu_quota_us == policy.cpu_quota_us()
            && self.cpu_period_us == CPU_PERIOD_US
            && self.memory_high_bytes == policy.memory_high_bytes
            && self.memory_max_bytes == policy.memory_max_bytes
            && self.memory_swap_max_bytes == 0
            && self.pids_max == u64::from(policy.pids_max)
    }
}

/// No Deserialize/Clone or public success constructor. An open directory is
/// retained independently of its path. Dropping it never deletes the cgroup.
pub struct RoventTrainingResources {
    delegated: File,
    delegated_path: PathBuf,
    delegated_identity: DirectoryIdentity,
    leaf: File,
    leaf_path: PathBuf,
    leaf_identity: DirectoryIdentity,
    membership: PathBuf,
    policy: RoventTrainingResourcePolicy,
    prepared_readback: RoventTrainingResourceReadback,
    available_controllers: Vec<String>,
    subtree_controllers: Vec<String>,
}
impl std::fmt::Debug for RoventTrainingResources {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RoventTrainingResources")
            .field("leaf_path", &self.leaf_path)
            .field("leaf_identity", &self.leaf_identity)
            .field("membership", &self.membership)
            .field("policy", &self.policy)
            .field("prepared_readback", &self.prepared_readback)
            .finish_non_exhaustive()
    }
}

/// Partial setup retains both the original delegation and any opened leaf.
/// The directory is never deleted or a child killed as error compensation.
pub struct RoventTrainingResourceFailure {
    pub error: ArenaError,
    delegated: File,
    delegated_path: PathBuf,
    leaf_name: String,
    opened_leaf_directory: Option<Dir>,
    leaf: Option<File>,
    leaf_created: bool,
    policy: RoventTrainingResourcePolicy,
}
impl std::fmt::Debug for RoventTrainingResourceFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RoventTrainingResourceFailure")
            .field("error", &self.error)
            .field("delegated_path", &self.delegated_path)
            .field("leaf_name", &self.leaf_name)
            .field("leaf_created", &self.leaf_created)
            .field(
                "leaf_descriptor_retained",
                &(self.leaf.is_some() || self.opened_leaf_directory.is_some()),
            )
            .finish_non_exhaustive()
    }
}
impl RoventTrainingResourceFailure {
    pub fn delegated_file(&self) -> &File {
        &self.delegated
    }
    pub fn leaf_file(&self) -> Option<&File> {
        self.leaf.as_ref()
    }
    pub fn leaf_directory(&self) -> Option<&Dir> {
        self.opened_leaf_directory.as_ref()
    }
    pub fn observation(&self) -> Value {
        serde_json::json!({
            "domain":"rz-rovent-training-resource-failure/1",
            "delegated_path":self.delegated_path,"leaf_name":self.leaf_name,
            "leaf_created":self.leaf_created,
            "leaf_descriptor_retained":self.leaf.is_some() || self.opened_leaf_directory.is_some(),
            "policy":self.policy,"configured":false,"automatic_cleanup":false
        })
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct RoventTrainingMembershipObservation {
    pub pid: u32,
    pub process_start_ticks: u64,
    pub membership: PathBuf,
    pub thread_ids: Vec<u32>,
    pub affinity: Vec<u32>,
    pub readback: RoventTrainingResourceReadback,
    pub scope: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct RoventTrainingClosedObservation {
    pub processes_empty: bool,
    pub populated: bool,
    pub memory_peak_bytes: u64,
    pub readback: RoventTrainingResourceReadback,
    pub automatic_cleanup: bool,
    pub scope: &'static str,
}

/// A kernel limit reading. Unlimited is an observation, not a numeric zero.
#[derive(Clone, Copy, Debug, Serialize, Eq, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RoventTrainingScopeLimit {
    Numeric { value: u64 },
    Unlimited,
}
impl RoventTrainingScopeLimit {
    fn parse(text: &str) -> Result<Self, ArenaError> {
        match text.trim() {
            "max" => Ok(Self::Unlimited),
            number => Ok(Self::Numeric {
                value: canonical_u64(number)?,
            }),
        }
    }
    fn equals(&self, expected: u64) -> bool {
        matches!(self, Self::Numeric { value } if *value == expected)
    }
}

#[derive(Clone, Debug, Serialize, Eq, PartialEq)]
pub struct RoventTrainingTotalScopeReadback {
    pub cpu_quota_us: RoventTrainingScopeLimit,
    pub cpu_period_us: u64,
    pub memory_high_bytes: RoventTrainingScopeLimit,
    pub memory_max_bytes: RoventTrainingScopeLimit,
    pub memory_swap_max_bytes: RoventTrainingScopeLimit,
    pub pids_max: RoventTrainingScopeLimit,
}
impl RoventTrainingTotalScopeReadback {
    fn read(directory: &Dir) -> Result<Self, ArenaError> {
        let raw = read_kernel(directory, "cpu.max", 128, false)?;
        let mut fields = raw.split_whitespace();
        let quota = fields
            .next()
            .ok_or_else(|| invalid("total scope CPU quota absent"))?;
        let period = fields
            .next()
            .ok_or_else(|| invalid("total scope CPU period absent"))?;
        check(fields.next().is_none(), "total scope CPU field count")?;
        let period = canonical_u64(period)?;
        check(period > 0, "total scope CPU period is zero")?;
        let limit =
            |name| RoventTrainingScopeLimit::parse(&read_kernel(directory, name, 128, false)?);
        Ok(Self {
            cpu_quota_us: RoventTrainingScopeLimit::parse(quota)?,
            cpu_period_us: period,
            memory_high_bytes: limit("memory.high")?,
            memory_max_bytes: limit("memory.max")?,
            memory_swap_max_bytes: limit("memory.swap.max")?,
            pids_max: limit("pids.max")?,
        })
    }
    fn matches(&self, policy: &RoventTrainingResourcePolicy) -> bool {
        let cpu_matches = match self.cpu_quota_us {
            RoventTrainingScopeLimit::Numeric { value } => {
                value > 0
                    && self.cpu_period_us > 0
                    && u128::from(value) * 1000
                        == u128::from(policy.cpu_quota_millicpus) * u128::from(self.cpu_period_us)
            }
            RoventTrainingScopeLimit::Unlimited => false,
        };
        cpu_matches
            && self.memory_high_bytes.equals(policy.memory_high_bytes)
            && self.memory_max_bytes.equals(policy.memory_max_bytes)
            && self.memory_swap_max_bytes.equals(0)
            && self.pids_max.equals(u64::from(policy.pids_max))
    }
}

/// The held delegated ancestor's ceiling includes the original parent and leaf.
/// This neither proves exclusive use nor excludes contention from other work.
/// No caller-created JSON can turn this snapshot into a process-lifetime proof.
#[derive(Clone, Debug, Serialize)]
pub struct RoventTrainingTotalParentScopeObservation {
    pub verified: bool,
    pub unknown_reason: Option<String>,
    pub parent_pid: u32,
    pub parent_membership: Option<PathBuf>,
    pub common_ancestor_path: PathBuf,
    pub common_ancestor_membership: Option<PathBuf>,
    pub common_ancestor_device: u64,
    pub common_ancestor_inode: u64,
    pub child_membership: PathBuf,
    pub declared_policy: RoventTrainingResourcePolicy,
    pub readback: Option<RoventTrainingTotalScopeReadback>,
    /// Direct processes only. Descendant processes are not enumerated here.
    pub ancestor_direct_processes: Option<Vec<u32>>,
    pub scope: &'static str,
}

impl RoventTrainingResources {
    pub fn prepare(
        delegated: File,
        path: &Path,
        leaf_name: &str,
        policy: RoventTrainingResourcePolicy,
    ) -> Result<Self, Box<RoventTrainingResourceFailure>> {
        let mut failure = RoventTrainingResourceFailure {
            error: invalid("resource preparation incomplete"),
            delegated,
            delegated_path: path.to_path_buf(),
            leaf_name: leaf_name.into(),
            opened_leaf_directory: None,
            leaf: None,
            leaf_created: false,
            policy,
        };
        match Self::prepare_inner(&mut failure) {
            Ok(owner) => Ok(owner),
            Err(error) => {
                failure.error = error;
                Err(Box::new(failure))
            }
        }
    }
    fn prepare_inner(p: &mut RoventTrainingResourceFailure) -> Result<Self, ArenaError> {
        p.policy.validate()?;
        validate_absolute(&p.delegated_path)?;
        check(
            valid_leaf(&p.leaf_name),
            "leaf must be a unique rovent-* basename",
        )?;
        let leaf_path = p.delegated_path.join(&p.leaf_name);
        validate_absolute(&leaf_path)?;
        ensure_cgroup(&p.delegated)?;
        let delegated_identity = DirectoryIdentity::of(&p.delegated)?;
        check(
            p.delegated
                .metadata()
                .map_err(|_| io("delegated_owner"))?
                .uid()
                == Uid::effective().as_raw(),
            "delegated cgroup directory must belong to the current owner",
        )?;
        let named = open_absolute(&p.delegated_path)?;
        let named_file = directory_file(&named)?;
        ensure_cgroup(&named_file)?;
        check(
            DirectoryIdentity::of(&named_file)? == delegated_identity,
            "delegated path and original descriptor differ",
        )?;
        let available = controller_names(&read_kernel(&named, "cgroup.controllers", 4096, false)?)?;
        let subtree =
            controller_names(&read_kernel(&named, "cgroup.subtree_control", 4096, false)?)?;
        check(
            ["cpu", "memory", "pids"]
                .iter()
                .all(|c| available.iter().any(|s| s == c) && subtree.iter().any(|s| s == c)),
            "cpu/memory/pids must already be delegated; no subtree_control changes are made",
        )?;
        let proc_root = proc_root()?;
        let self_dir = proc_root
            .open_dir_nofollow(std::process::id().to_string())
            .map_err(|_| io("self_proc_open"))?;
        let mountinfo = read_kernel(&self_dir, "mountinfo", MAX_MOUNTINFO, true)?;
        let parent_membership =
            parse_membership(&read_kernel(&self_dir, "cgroup", MAX_PROC_TEXT, true)?)?;
        let delegated_membership = membership_for_path(&mountinfo, &p.delegated_path)?;
        let membership = delegated_membership.join(&p.leaf_name);
        validate_absolute(&membership)?;
        check(
            related_memberships(&parent_membership, &delegated_membership),
            "delegated directory is a sibling or outside the parent's cgroup tree",
        )?;
        // mkdirat is exclusive and applies 0700 atomically. Never reuse an
        // existing leaf and never write any ancestor control file.
        mkdirat(&p.delegated, p.leaf_name.as_str(), Mode::S_IRWXU)
            .map_err(|_| io("exclusive_leaf_create"))?;
        p.leaf_created = true;
        p.opened_leaf_directory = Some(
            named
                .open_dir_nofollow(&p.leaf_name)
                .map_err(|_| io("leaf_open"))?,
        );
        // Retain the first successful directory open before any additional FD
        // allocation can fail (for example EMFILE during directory_file).
        let leaf_dir = p
            .opened_leaf_directory
            .as_ref()
            .expect("retained first leaf directory");
        p.leaf = Some(directory_file(leaf_dir)?);
        let leaf = p.leaf.as_ref().expect("retained just opened leaf");
        ensure_cgroup(leaf)?;
        let leaf_identity = DirectoryIdentity::of(leaf)?;
        let meta = leaf.metadata().map_err(|_| io("leaf_metadata"))?;
        check(
            meta.uid() == Uid::effective().as_raw() && meta.mode() & 0o7777 == 0o700,
            "fresh cgroup leaf must be current-owner 0700",
        )?;
        check(
            read_pids(leaf_dir)?.is_empty(),
            "fresh leaf already has processes",
        )?;
        for (name, value) in [
            (
                "cpu.max",
                format!("{} {}", p.policy.cpu_quota_us(), CPU_PERIOD_US),
            ),
            ("memory.high", p.policy.memory_high_bytes.to_string()),
            ("memory.max", p.policy.memory_max_bytes.to_string()),
            ("memory.swap.max", "0".into()),
            ("pids.max", p.policy.pids_max.to_string()),
        ] {
            write_kernel(leaf_dir, name, &value)?;
        }
        let readback = RoventTrainingResourceReadback::read(leaf_dir)?;
        check(
            readback.matches(&p.policy),
            "actual cgroup limits differ after writing",
        )?;
        let owner = Self {
            delegated: p
                .delegated
                .try_clone()
                .map_err(|_| io("delegation_clone"))?,
            delegated_path: p.delegated_path.clone(),
            delegated_identity,
            leaf: leaf.try_clone().map_err(|_| io("leaf_clone"))?,
            leaf_path,
            leaf_identity,
            membership,
            policy: p.policy.clone(),
            prepared_readback: readback,
            available_controllers: available,
            subtree_controllers: subtree,
        };
        owner.recheck_identity()?;
        Ok(owner)
    }
    pub fn root_file(&self) -> &File {
        &self.leaf
    }
    pub fn policy(&self) -> &RoventTrainingResourcePolicy {
        &self.policy
    }
    /// Reads only the original held common delegated ancestor. If it is below
    /// the current parent, unlimited or otherwise unverified, keep Unknown;
    /// never search arbitrary higher nodes or alter any configured resource.
    pub fn observe_total_parent_scope(
        &self,
    ) -> Result<RoventTrainingTotalParentScopeObservation, ArenaError> {
        let mut observation = RoventTrainingTotalParentScopeObservation {
            verified: false,
            unknown_reason: None,
            parent_pid: std::process::id(),
            parent_membership: None,
            common_ancestor_path: self.delegated_path.clone(),
            common_ancestor_membership: None,
            common_ancestor_device: self.delegated_identity.device,
            common_ancestor_inode: self.delegated_identity.inode,
            child_membership: self.membership.clone(),
            declared_policy: self.policy.clone(),
            readback: None,
            ancestor_direct_processes: None,
            scope: "bounded_original_common_ancestor_total_ceiling_snapshot_not_exclusivity_contention_or_lifetime_proof",
        };
        if let Err(error) = self.observe_total_parent_scope_checked(&mut observation) {
            // Functional callers may preserve Unknown; strict parent admission
            // rejects it before spawn. Missing evidence is never invented.
            observation.unknown_reason = Some(error.to_string());
        } else {
            observation.verified = true;
        }
        Ok(observation)
    }
    fn observe_total_parent_scope_checked(
        &self,
        observation: &mut RoventTrainingTotalParentScopeObservation,
    ) -> Result<(), ArenaError> {
        self.recheck_identity()?;
        let proc_root = proc_root()?;
        let parent = proc_root
            .open_dir_nofollow(std::process::id().to_string())
            .map_err(|_| io("total_scope_parent_proc"))?;
        let membership = parse_membership(&read_kernel(&parent, "cgroup", MAX_PROC_TEXT, true)?)?;
        observation.parent_membership = Some(membership.clone());
        let mounts = read_kernel(&parent, "mountinfo", MAX_MOUNTINFO, true)?;
        let common = membership_for_path(&mounts, &self.delegated_path)?;
        observation.common_ancestor_membership = Some(common.clone());
        check(
            membership.starts_with(&common),
            "total scope unknown: parent is outside the held delegated ancestor",
        )?;
        check(
            self.membership.starts_with(&common)
                && membership_for_path(&mounts, &self.leaf_path)? == self.membership,
            "total scope unknown: child leaf and held ancestor mount mapping differ",
        )?;
        let directory = Dir::from_std_file(
            self.delegated
                .try_clone()
                .map_err(|_| io("total_scope_delegated_clone"))?,
        );
        let direct_processes = read_pids(&directory)?;
        observation.ancestor_direct_processes = Some(direct_processes.iter().copied().collect());
        check(
            membership != common || direct_processes.contains(&std::process::id()),
            "total scope unknown: direct parent membership changed",
        )?;
        let readback = RoventTrainingTotalScopeReadback::read(&directory)?;
        observation.readback = Some(readback.clone());
        check(
            !matches!(readback.cpu_quota_us, RoventTrainingScopeLimit::Unlimited),
            "total scope unknown: common ancestor CPU quota is unlimited",
        )?;
        check(
            readback.matches(&self.policy),
            "total scope unknown: common ancestor limits do not exactly match policy",
        )?;
        check(
            parse_membership(&read_kernel(&parent, "cgroup", MAX_PROC_TEXT, true)?)? == membership
                && RoventTrainingTotalScopeReadback::read(&directory)? == readback,
            "total scope unknown: membership or limits changed during snapshot",
        )?;
        let mounts_after = read_kernel(&parent, "mountinfo", MAX_MOUNTINFO, true)?;
        check(
            membership_for_path(&mounts_after, &self.delegated_path)? == common
                && membership_for_path(&mounts_after, &self.leaf_path)? == self.membership,
            "total scope unknown: cgroup namespace or mount mapping changed",
        )?;
        self.recheck_identity()?;
        Ok(())
    }
    pub fn startup_json(&self) -> Result<Value, ArenaError> {
        let directory = self.recheck_identity()?;
        let readback = RoventTrainingResourceReadback::read(&directory)?;
        check(
            readback.matches(&self.policy),
            "resource settings changed before startup",
        )?;
        Ok(serde_json::json!({
            "domain":"rz-rovent-training-resource-startup/1",
            "migration":"child_self_only_before_torch",
            "path":self.leaf_path,"device":self.leaf_identity.device,"inode":self.leaf_identity.inode,
            "membership":self.membership,"policy":self.policy,
            "prepared_readback":self.prepared_readback,"readback":readback,
            "available_controllers_snapshot":self.available_controllers,
            "subtree_controllers_snapshot":self.subtree_controllers,
            "scope":"configured_leaf_readback_not_child_membership_or_lifetime_proof"
        }))
    }
    pub fn verify_child_membership(
        &self,
        pid: u32,
    ) -> Result<RoventTrainingMembershipObservation, ArenaError> {
        check(
            pid > 0 && pid != std::process::id() && i32::try_from(pid).is_ok(),
            "membership check requires an explicit child PID",
        )?;
        let leaf = self.recheck_identity()?;
        let root = proc_root()?;
        let child = root
            .open_dir_nofollow(pid.to_string())
            .map_err(|_| io("child_proc_open"))?;
        let start = process_start(&read_kernel(&child, "stat", MAX_PROC_TEXT, true)?, pid)?;
        let member = parse_membership(&read_kernel(&child, "cgroup", MAX_PROC_TEXT, true)?)?;
        check(
            member == self.membership,
            "child is not in the configured leaf",
        )?;
        let procs = read_pids(&leaf)?;
        check(
            procs == BTreeSet::from([pid]),
            "trainer leaf contains another or no process",
        )?;
        let tasks = child
            .open_dir_nofollow("task")
            .map_err(|_| io("task_open"))?;
        let thread_ids = task_ids(&tasks, self.policy.pids_max)?;
        let mut expected = CpuSet::new();
        for &cpu in &self.policy.affinity {
            expected
                .set(cpu as usize)
                .map_err(|_| invalid("CPU affinity is not representable"))?;
        }
        for &tid in &thread_ids {
            let task = tasks
                .open_dir_nofollow(tid.to_string())
                .map_err(|_| io("task_identity_open"))?;
            check(
                parse_membership(&read_kernel(&task, "cgroup", MAX_PROC_TEXT, true)?)?
                    == self.membership,
                "trainer thread belongs to another cgroup",
            )?;
            let actual = sched_getaffinity(Pid::from_raw(
                i32::try_from(tid).map_err(|_| invalid("thread PID overflow"))?,
            ))
            .map_err(|_| io("thread_affinity"))?;
            check(
                actual == expected,
                "trainer thread affinity differs from policy",
            )?;
        }
        let readback = RoventTrainingResourceReadback::read(&leaf)?;
        check(
            readback.matches(&self.policy),
            "resource settings changed during child admission",
        )?;
        check(
            task_ids(&tasks, self.policy.pids_max)? == thread_ids
                && process_start(&read_kernel(&child, "stat", MAX_PROC_TEXT, true)?, pid)? == start
                && parse_membership(&read_kernel(&child, "cgroup", MAX_PROC_TEXT, true)?)?
                    == member
                && read_pids(&leaf)? == procs,
            "child identity/membership/thread snapshot changed during admission",
        )?;
        self.recheck_identity()?;
        Ok(RoventTrainingMembershipObservation {
            pid,
            process_start_ticks: start,
            membership: member,
            thread_ids,
            affinity: self.policy.affinity.clone(),
            readback,
            scope: "bounded_current_process_and_thread_snapshot_not_lifetime_enforcement",
        })
    }
    /// A finite post-closure snapshot. The caller must separately retain and
    /// verify original process/group closure. This never removes the leaf.
    pub fn observe_closed(&self) -> Result<RoventTrainingClosedObservation, ArenaError> {
        let leaf = self.recheck_identity()?;
        check(
            read_pids(&leaf)?.is_empty(),
            "trainer leaf still contains processes",
        )?;
        let populated = populated(&read_kernel(&leaf, "cgroup.events", 4096, false)?)?;
        check(
            !populated,
            "trainer cgroup or descendants are still populated",
        )?;
        let peak = read_number(&leaf, "memory.peak")?;
        let readback = RoventTrainingResourceReadback::read(&leaf)?;
        check(
            readback.matches(&self.policy),
            "closed resource readback differs from policy",
        )?;
        check(
            read_pids(&leaf)?.is_empty() && !populated_value(&leaf)?,
            "trainer cgroup changed during closure snapshot",
        )?;
        self.recheck_identity()?;
        Ok(RoventTrainingClosedObservation {
            processes_empty: true,
            populated: false,
            memory_peak_bytes: peak,
            readback,
            automatic_cleanup: false,
            scope: "empty_at_readback_after_separate_process_closure_not_future_lifetime_guarantee",
        })
    }
    fn recheck_identity(&self) -> Result<Dir, ArenaError> {
        ensure_cgroup(&self.delegated)?;
        ensure_cgroup(&self.leaf)?;
        check(
            DirectoryIdentity::of(&self.delegated)? == self.delegated_identity
                && DirectoryIdentity::of(&self.leaf)? == self.leaf_identity
                && self
                    .delegated
                    .metadata()
                    .map_err(|_| io("delegated_owner_recheck"))?
                    .uid()
                    == Uid::effective().as_raw(),
            "retained cgroup directory identity changed",
        )?;
        let delegated = open_absolute(&self.delegated_path)?;
        check(
            DirectoryIdentity::of(&directory_file(&delegated)?)? == self.delegated_identity,
            "named delegated cgroup changed",
        )?;
        let name = self
            .leaf_path
            .file_name()
            .ok_or_else(|| invalid("leaf basename absent"))?;
        let leaf = delegated
            .open_dir_nofollow(name)
            .map_err(|_| io("leaf_named_recheck"))?;
        let file = directory_file(&leaf)?;
        let meta = file.metadata().map_err(|_| io("leaf_recheck_metadata"))?;
        check(
            DirectoryIdentity::of(&file)? == self.leaf_identity
                && meta.uid() == Uid::effective().as_raw()
                && meta.mode() & 0o7777 == 0o700,
            "named leaf identity/owner/mode changed",
        )?;
        Ok(leaf)
    }
}

fn invalid(detail: &str) -> ArenaError {
    ArenaError::Invalid(format!("rovent_training_resources.{detail}"))
}
fn io(stage: &str) -> ArenaError {
    ArenaError::Io(format!("rovent_training_resources.{stage}"))
}
fn check(ok: bool, detail: &str) -> Result<(), ArenaError> {
    if ok { Ok(()) } else { Err(invalid(detail)) }
}
fn valid_leaf(name: &str) -> bool {
    name.starts_with("rovent-")
        && (8..=96).contains(&name.len())
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b))
}
fn validate_absolute(path: &Path) -> Result<(), ArenaError> {
    let text = path
        .to_str()
        .ok_or_else(|| invalid("resource path is not UTF-8"))?;
    let mut canonical = PathBuf::from("/");
    check(
        path.is_absolute() && text.len() <= MAX_PATH && !text.chars().any(char::is_control),
        "resource path must be bounded and absolute",
    )?;
    for part in path.components() {
        match part {
            Component::RootDir => (),
            Component::Normal(name) => canonical.push(name),
            _ => return Err(invalid("resource path contains dot/traversal/prefix")),
        }
    }
    check(
        canonical.as_os_str() == path.as_os_str(),
        "resource path must have canonical lexical spelling",
    )
}
fn open_absolute(path: &Path) -> Result<Dir, ArenaError> {
    validate_absolute(path)?;
    let mut directory =
        Dir::open_ambient_dir("/", ambient_authority()).map_err(|_| io("root_open"))?;
    for part in path.components() {
        if let Component::Normal(name) = part {
            directory = directory
                .open_dir_nofollow(name)
                .map_err(|_| io("absolute_nofollow"))?;
        }
    }
    Ok(directory)
}
fn directory_file(directory: &Dir) -> Result<File, ArenaError> {
    let mut options = cap_std::fs::OpenOptions::new();
    options.read(true).follow(FollowSymlinks::No);
    directory
        .open_with(".", &options)
        .map(|f| f.into_std())
        .map_err(|_| io("directory_fd"))
}
fn ensure_cgroup(file: &File) -> Result<(), ArenaError> {
    check(
        fstatfs(file)
            .map_err(|_| io("cgroup_statfs"))?
            .filesystem_type()
            == CGROUP2_SUPER_MAGIC,
        "resource descriptor is not cgroup v2",
    )
}
fn proc_root() -> Result<Dir, ArenaError> {
    let root = open_absolute(Path::new("/proc"))?;
    check(
        fstatfs(&directory_file(&root)?)
            .map_err(|_| io("proc_statfs"))?
            .filesystem_type()
            == PROC_SUPER_MAGIC,
        "proc directory is not procfs",
    )?;
    Ok(root)
}
fn read_kernel(
    directory: &Dir,
    name: &str,
    maximum: usize,
    procfs: bool,
) -> Result<String, ArenaError> {
    check(
        !name.is_empty()
            && name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"._".contains(&b)),
        "kernel read requires a fixed basename",
    )?;
    let mut options = cap_std::fs::OpenOptions::new();
    options.read(true).follow(FollowSymlinks::No);
    let mut file = directory
        .open_with(name, &options)
        .map_err(|_| io("kernel_open"))?;
    let expected = if procfs {
        PROC_SUPER_MAGIC
    } else {
        CGROUP2_SUPER_MAGIC
    };
    check(
        fstatfs(&file)
            .map_err(|_| io("kernel_statfs"))?
            .filesystem_type()
            == expected
            && file
                .metadata()
                .map_err(|_| io("kernel_metadata"))?
                .is_file(),
        "kernel read filesystem/type differs",
    )?;
    let mut raw = Vec::with_capacity(maximum.min(4096));
    (&mut file)
        .take((maximum + 1) as u64)
        .read_to_end(&mut raw)
        .map_err(|_| io("kernel_read"))?;
    check(raw.len() <= maximum, "kernel read byte limit exceeded")?;
    String::from_utf8(raw).map_err(|_| invalid("kernel read is not UTF-8"))
}
fn write_kernel(directory: &Dir, name: &str, value: &str) -> Result<(), ArenaError> {
    check(
        [
            "cpu.max",
            "memory.high",
            "memory.max",
            "memory.swap.max",
            "pids.max",
        ]
        .contains(&name),
        "only fresh leaf limit files may be written",
    )?;
    write_cgroup_value(directory, name, value)
}
// The limit-file whitelist above remains unchanged. The native child module
// may additionally write only its own actual PID to the retained fresh leaf's
// cgroup.procs; both paths use this same bounded no-follow cgroup2 write.
fn write_cgroup_value(directory: &Dir, name: &str, value: &str) -> Result<(), ArenaError> {
    let mut options = cap_std::fs::OpenOptions::new();
    options.write(true).follow(FollowSymlinks::No);
    let mut file = directory
        .open_with(name, &options)
        .map_err(|_| io("control_open"))?;
    check(
        fstatfs(&file)
            .map_err(|_| io("control_statfs"))?
            .filesystem_type()
            == CGROUP2_SUPER_MAGIC
            && file
                .metadata()
                .map_err(|_| io("control_metadata"))?
                .is_file(),
        "control write is not a regular cgroup v2 file",
    )?;
    let raw = format!("{value}\n");
    let written = file
        .write(raw.as_bytes())
        .map_err(|_| io("control_write"))?;
    check(written == raw.len(), "partial cgroup control write")
}
fn canonical_u64(text: &str) -> Result<u64, ArenaError> {
    check(
        !text.is_empty()
            && text.len() <= 20
            && text.bytes().all(|b| b.is_ascii_digit())
            && (text == "0" || !text.starts_with('0')),
        "noncanonical numeric kernel value",
    )?;
    text.parse()
        .map_err(|_| invalid("numeric kernel value overflow"))
}
fn read_number(directory: &Dir, name: &str) -> Result<u64, ArenaError> {
    canonical_u64(read_kernel(directory, name, 128, false)?.trim())
}
fn controller_names(text: &str) -> Result<Vec<String>, ArenaError> {
    let mut names = BTreeSet::new();
    for name in text.split_whitespace() {
        check(
            name.len() <= 64
                && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
                && names.insert(name.to_owned()),
            "invalid controller snapshot",
        )?;
    }
    Ok(names.into_iter().collect())
}
fn read_pids(directory: &Dir) -> Result<BTreeSet<u32>, ArenaError> {
    let mut pids = BTreeSet::new();
    for line in read_kernel(directory, "cgroup.procs", MAX_CGROUP_PROCS, false)?.lines() {
        let number = canonical_u64(line)?;
        let pid = u32::try_from(number).map_err(|_| invalid("cgroup PID overflow"))?;
        check(
            pid > 0 && i32::try_from(pid).is_ok() && pids.insert(pid) && pids.len() <= 4096,
            "invalid/duplicate or excessive cgroup processes",
        )?;
    }
    Ok(pids)
}
fn parse_membership(text: &str) -> Result<PathBuf, ArenaError> {
    let mut value = None;
    for line in text.lines() {
        if let Some(path) = line.strip_prefix("0::") {
            check(value.is_none(), "duplicate unified cgroup membership")?;
            let path = PathBuf::from(path);
            validate_absolute(&path)?;
            value = Some(path);
        }
    }
    value.ok_or_else(|| invalid("unified cgroup membership absent"))
}
fn related_memberships(parent: &Path, delegated: &Path) -> bool {
    parent.starts_with(delegated) || delegated.starts_with(parent)
}
fn mount_field(text: &str) -> Result<PathBuf, ArenaError> {
    let raw = text.as_bytes();
    let mut decoded = Vec::with_capacity(raw.len());
    let mut i = 0;
    while i < raw.len() {
        if raw[i] == b'\\' {
            check(
                i + 3 < raw.len() && raw[i + 1..=i + 3].iter().all(|b| (b'0'..=b'7').contains(b)),
                "invalid mountinfo escape",
            )?;
            let n = (raw[i + 1] - b'0') as u16 * 64
                + (raw[i + 2] - b'0') as u16 * 8
                + (raw[i + 3] - b'0') as u16;
            check(n <= 255 && n != 0, "invalid mountinfo escaped byte")?;
            decoded.push(n as u8);
            i += 4;
        } else {
            decoded.push(raw[i]);
            i += 1;
        }
    }
    let path =
        PathBuf::from(String::from_utf8(decoded).map_err(|_| invalid("mount path encoding"))?);
    validate_absolute(&path)?;
    Ok(path)
}
fn membership_for_path(mountinfo: &str, path: &Path) -> Result<PathBuf, ArenaError> {
    let mut selected: Option<(PathBuf, PathBuf)> = None;
    for line in mountinfo.lines() {
        let Some((left, right)) = line.split_once(" - ") else {
            return Err(invalid("mountinfo separator absent"));
        };
        if right.split_whitespace().next() != Some("cgroup2") {
            continue;
        }
        let fields: Vec<_> = left.split_whitespace().collect();
        check(fields.len() >= 6, "mountinfo cgroup field count")?;
        let root = mount_field(fields[3])?;
        let point = mount_field(fields[4])?;
        if path.starts_with(&point)
            && selected
                .as_ref()
                .is_none_or(|(p, _)| point.as_os_str().len() > p.as_os_str().len())
        {
            selected = Some((point, root));
        }
    }
    let (point, root) =
        selected.ok_or_else(|| invalid("delegated path has no cgroup2 mount mapping"))?;
    let relative = path
        .strip_prefix(&point)
        .map_err(|_| invalid("mount path prefix"))?;
    let membership = root.join(relative);
    validate_absolute(&membership)?;
    Ok(membership)
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ProcessIdentity {
    parent_pid: u32,
    start_ticks: u64,
}
/// Parse original proc identity only. A self/root PID need not be a child of
/// this caller; each consumer verifies its own actual relationship separately.
fn parse_process_identity(text: &str, pid: u32) -> Result<ProcessIdentity, ArenaError> {
    let begin = text
        .find('(')
        .ok_or_else(|| invalid("proc stat command absent"))?;
    let end = text
        .rfind(") ")
        .ok_or_else(|| invalid("proc stat command terminator absent"))?;
    check(
        text[..begin].trim() == pid.to_string() && end >= begin,
        "proc stat PID differs",
    )?;
    let fields: Vec<_> = text[end + 2..].split_whitespace().collect();
    let parent = canonical_u64(
        fields
            .get(1)
            .ok_or_else(|| invalid("proc stat parent absent"))?,
    )?;
    check(parent <= i32::MAX as u64, "proc stat parent PID overflow")?;
    let start_ticks = canonical_u64(
        fields
            .get(19)
            .ok_or_else(|| invalid("proc stat start time absent"))?,
    )?;
    Ok(ProcessIdentity {
        parent_pid: parent as u32,
        start_ticks,
    })
}
fn require_process_parent(identity: &ProcessIdentity, parent_pid: u32) -> Result<(), ArenaError> {
    check(
        identity.parent_pid == parent_pid,
        "PID is not a direct child of the original parent",
    )
}
/// The trainer path still requires a direct child of this original caller.
fn process_start(text: &str, pid: u32) -> Result<u64, ArenaError> {
    let identity = parse_process_identity(text, pid)?;
    require_process_parent(&identity, std::process::id())?;
    Ok(identity.start_ticks)
}
fn task_ids(tasks: &Dir, maximum: u32) -> Result<Vec<u32>, ArenaError> {
    let mut ids = BTreeSet::new();
    for entry in tasks.entries().map_err(|_| io("task_enumeration"))? {
        let entry = entry.map_err(|_| io("task_entry"))?;
        let name = entry.file_name();
        let name = name.to_str().ok_or_else(|| invalid("task ID encoding"))?;
        let tid = u32::try_from(canonical_u64(name)?).map_err(|_| invalid("task ID overflow"))?;
        check(
            tid > 0
                && i32::try_from(tid).is_ok()
                && ids.insert(tid)
                && ids.len() <= maximum as usize,
            "invalid/duplicate or excessive task IDs",
        )?;
    }
    check(!ids.is_empty(), "trainer has no observed tasks")?;
    Ok(ids.into_iter().collect())
}
fn populated(text: &str) -> Result<bool, ArenaError> {
    let mut answer = None;
    for line in text.lines() {
        if let Some(value) = line.strip_prefix("populated ") {
            check(
                answer.is_none() && (value == "0" || value == "1"),
                "invalid populated snapshot",
            )?;
            answer = Some(value == "1");
        }
    }
    answer.ok_or_else(|| invalid("cgroup populated observation absent"))
}
fn populated_value(directory: &Dir) -> Result<bool, ArenaError> {
    populated(&read_kernel(directory, "cgroup.events", 4096, false)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn policy() -> RoventTrainingResourcePolicy {
        RoventTrainingResourcePolicy {
            cpu_quota_millicpus: 2000,
            memory_high_bytes: 6 << 30,
            memory_max_bytes: 12 << 30,
            pids_max: 128,
            affinity: vec![0, 2],
            device: "cuda:0".into(),
        }
    }
    fn proc_stat(pid: u32, parent: u32, start: u64) -> String {
        let mut fields = vec!["0".to_owned(); 20];
        fields[0] = "R".into();
        fields[1] = parent.to_string();
        fields[19] = start.to_string();
        format!("{pid} (native ) worker) {}", fields.join(" "))
    }
    #[test]
    fn proc_identity_accepts_original_self_and_descendants_without_inventing_parentage() {
        let original = parse_process_identity(&proc_stat(42, 7, 123), 42).unwrap();
        assert_eq!(original.parent_pid, 7);
        assert_eq!(original.start_ticks, 123);
        assert!(parse_process_identity(&proc_stat(42, 7, 123), 43).is_err());
        let descendant = parse_process_identity(&proc_stat(84, 42, 456), 84).unwrap();
        assert!(require_process_parent(&descendant, 42).is_ok());
        assert!(require_process_parent(&descendant, 7).is_err());
        assert!(parse_process_identity("42 (native) R 7", 42).is_err());
    }
    #[test]
    fn trainer_start_still_requires_the_original_callers_direct_child() {
        let caller = std::process::id();
        let other_parent = if caller == 1 { 2 } else { 1 };
        assert_eq!(process_start(&proc_stat(84, caller, 456), 84).unwrap(), 456);
        assert!(process_start(&proc_stat(84, other_parent, 456), 84).is_err());
        assert!(parse_process_identity(&proc_stat(caller, other_parent, 123), caller).is_ok());
        assert!(process_start(&proc_stat(caller, other_parent, 123), caller).is_err());
    }
    #[test]
    fn policy_is_explicit_bounded_and_duplicate_fields_are_rejected() {
        let p = policy();
        let raw = serde_json::to_string(&p).unwrap();
        assert_eq!(RoventTrainingResourcePolicy::from_json(&raw).unwrap(), p);
        assert!(
            RoventTrainingResourcePolicy::from_json(&raw.replacen(
                "\"pids_max\":",
                "\"extra\":0,\"pids_max\":",
                1
            ))
            .is_err()
        );
        assert!(
            RoventTrainingResourcePolicy::from_json(&raw.replacen(
                "\"pids_max\":",
                "\"pids_max\":1,\"pids_max\":",
                1
            ))
            .is_err()
        );
        for device in ["cuda", "cuda:-1", "cuda:00", "cuda:64", "gpu", " cpu"] {
            let mut wrong = p.clone();
            wrong.device = device.into();
            assert!(wrong.validate().is_err());
        }
        let mut wrong = p.clone();
        wrong.affinity = vec![0, 0];
        assert!(wrong.validate().is_err());
        let mut wrong = p.clone();
        wrong.affinity.clear();
        assert!(wrong.validate().is_err());
        let mut wrong = p.clone();
        wrong.memory_high_bytes = wrong.memory_max_bytes + 1;
        assert!(wrong.validate().is_err());
        let mut wrong = p.clone();
        wrong.cpu_quota_millicpus = 0;
        assert!(wrong.validate().is_err());
        assert_eq!(p.cpu_quota_us(), 2_000_000);
    }
    #[test]
    fn paths_mount_mapping_and_kernel_parsers_are_pure_and_fail_closed() {
        for path in [
            "relative",
            "/sys/../cg",
            "/sys/./cg",
            "/sys//cg",
            "/sys/cg/",
        ] {
            assert!(validate_absolute(Path::new(path)).is_err(), "{path}");
        }
        assert!(valid_leaf("rovent-job_1"));
        assert!(!valid_leaf("rovent-../job"));
        assert_eq!(
            parse_membership("0::/owner/job\n").unwrap(),
            Path::new("/owner/job")
        );
        assert!(parse_membership("0::/a\n0::/b\n").is_err());
        assert!(related_memberships(
            Path::new("/owner/task"),
            Path::new("/owner")
        ));
        assert!(related_memberships(
            Path::new("/owner"),
            Path::new("/owner/delegated")
        ));
        assert!(related_memberships(
            Path::new("/owner"),
            Path::new("/owner")
        ));
        assert!(!related_memberships(
            Path::new("/owner/a"),
            Path::new("/owner/b")
        ));
        assert!(!related_memberships(
            Path::new("/owner"),
            Path::new("/owner-other")
        ));
        let mounts = "31 20 0:29 /owner /sys/fs/cgroup rw - cgroup2 cgroup rw\n";
        assert_eq!(
            membership_for_path(mounts, Path::new("/sys/fs/cgroup/job")).unwrap(),
            Path::new("/owner/job")
        );
        assert!(membership_for_path(mounts, Path::new("/sys/fs/cgroup-other/job")).is_err());
        assert!(!populated("populated 0\nfrozen 0\n").unwrap());
        assert!(populated("populated 0\npopulated 0\n").is_err());
        assert!(canonical_u64("00").is_err());
        assert!(canonical_u64("max").is_err());
    }
    #[test]
    fn total_scope_readback_preserves_unlimited_and_exact_cpu_ratios() {
        let p = policy();
        let mut limits = RoventTrainingTotalScopeReadback {
            cpu_quota_us: RoventTrainingScopeLimit::Numeric { value: 200_000 },
            cpu_period_us: 100_000,
            memory_high_bytes: RoventTrainingScopeLimit::Numeric { value: 6 << 30 },
            memory_max_bytes: RoventTrainingScopeLimit::Numeric { value: 12 << 30 },
            memory_swap_max_bytes: RoventTrainingScopeLimit::Numeric { value: 0 },
            pids_max: RoventTrainingScopeLimit::Numeric { value: 128 },
        };
        assert!(limits.matches(&p));
        limits.cpu_quota_us = RoventTrainingScopeLimit::Unlimited;
        assert!(!limits.matches(&p));
        assert_eq!(
            RoventTrainingScopeLimit::parse("max\n").unwrap(),
            RoventTrainingScopeLimit::Unlimited
        );
        limits.cpu_quota_us = RoventTrainingScopeLimit::Numeric { value: 199_999 };
        assert!(!limits.matches(&p));
        limits.cpu_quota_us = RoventTrainingScopeLimit::Numeric { value: 200_000 };
        limits.memory_max_bytes = RoventTrainingScopeLimit::Unlimited;
        assert!(!limits.matches(&p));
    }
}
