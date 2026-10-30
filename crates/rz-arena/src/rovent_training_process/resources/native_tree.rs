//! The same delegated-cgroup owner, used before a native in-process launch.
//! This leaf does not load a model, spawn, stop, reap or delete a process. The
//! caller retains the original owner on every failure after partial migration.
use super::*;
use nix::sched::sched_setaffinity;
use std::collections::BTreeMap;

/// Current-process entry is issued only after the actual self migration and
/// readback. Numeric PID/path declarations cannot reconstruct this custody.
#[derive(Debug)]
pub struct RoventNativeProcessEntry {
    leaf: DirectoryIdentity,
    root_pid: u32,
    root_start_ticks: u64,
    original_membership: PathBuf,
    membership: PathBuf,
}

#[derive(Clone, Debug, Serialize, Eq, PartialEq)]
pub struct RoventNativeProcessSnapshot {
    pub pid: u32,
    pub start_ticks: u64,
    pub thread_ids: Vec<u32>,
    pub children: Vec<u32>,
    pub affinity: Vec<u32>,
}

/// A stable bounded pair of kernel snapshots, not continuous enforcement of
/// descendants which later escape or proof of exclusive CPU/topology use.
#[derive(Clone, Debug, Serialize)]
pub struct RoventNativeTreeObservation {
    pub root_pid: u32,
    pub root_start_ticks: u64,
    pub original_membership: PathBuf,
    pub membership: PathBuf,
    pub processes: Vec<RoventNativeProcessSnapshot>,
    pub kernel_threads: u64,
    pub readback: RoventTrainingResourceReadback,
    pub live_leaf_memory_peak_bytes: Option<u64>,
    pub live_leaf_memory_peak_unknown_reason: Option<String>,
    /// Only task directories confirmed absent after an unchanged process
    /// identity. Each successful retry still requires two matching full trees.
    pub exited_threads_reobserved: Vec<RoventExitedThreadObservation>,
    pub scope: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct RoventExitedThreadObservation {
    pub pid: u32,
    pub tid: u32,
    pub errno: Option<i32>,
}

enum NativeTreeAttemptError {
    ExitedThread(RoventExitedThreadObservation),
    Other(ArenaError),
}
impl From<ArenaError> for NativeTreeAttemptError {
    fn from(value: ArenaError) -> Self { Self::Other(value) }
}

fn task_open_failure(
    pid: u32, tid: u32, error: std::io::Error,
    recheck: impl FnOnce() -> Result<bool, ArenaError>,
) -> NativeTreeAttemptError {
    let errno = error.raw_os_error();
    let detail = || format!("rovent_training_resources.native_tree_thread pid={pid} tid={tid} kind={:?} errno={errno:?}", error.kind());
    if error.kind() == std::io::ErrorKind::NotFound {
        match recheck() {
            Ok(true) => return NativeTreeAttemptError::ExitedThread(RoventExitedThreadObservation { pid, tid, errno }),
            Ok(false) => {},
            Err(recheck_error) => return NativeTreeAttemptError::Other(ArenaError::Io(format!(
                "{}; identity recheck: {recheck_error}", detail()))),
        }
    }
    NativeTreeAttemptError::Other(ArenaError::Io(detail()))
}

fn retry_exited_threads<T>(mut attempt: impl FnMut() -> Result<T, NativeTreeAttemptError>)
    -> Result<(T, Vec<RoventExitedThreadObservation>), ArenaError> {
    let mut exited = Vec::new();
    for _ in 0..3 {
        match attempt() {
            Ok(value) => return Ok((value, exited)),
            Err(NativeTreeAttemptError::ExitedThread(detail)) => exited.push(detail),
            Err(NativeTreeAttemptError::Other(error)) => {
                if exited.is_empty() { return Err(error); }
                return Err(ArenaError::Io(format!("native tree failed after exited-thread observations {exited:?}: {error}")));
            }
        }
    }
    Err(ArenaError::Io(format!("native tree exited-thread reobservation exhausted: {exited:?}")))
}

impl RoventTrainingResources {
    /// Read-only original descriptor preparation shared by native upper
    /// callers. Reuse the same absolute no-follow path walk as the trainer;
    /// this issues no resource entry or configuration authority.
    pub fn open_native_delegation(path: &Path) -> Result<File, ArenaError> {
        validate_absolute(path)?;
        let directory = open_absolute(path)?;
        let file = directory_file(&directory)?;
        ensure_cgroup(&file)?;
        Ok(file)
    }

    /// Enter only this process before native/teacher/ORT threads are created.
    /// Existing ancestor controls are never written and Drop never migrates
    /// the caller back. On partial failure the same caller must keep this FD
    /// owner and its real process rather than claiming an untouched startup.
    pub fn enter_current_native_process(&self) -> Result<RoventNativeProcessEntry, ArenaError> {
        let leaf = self.recheck_identity()?;
        check(
            read_pids(&leaf)?.is_empty(),
            "native entry requires its fresh empty leaf",
        )?;
        let proc = proc_root()?;
        let pid = std::process::id();
        let current = proc
            .open_dir_nofollow(pid.to_string())
            .map_err(|_| io("native_self_proc"))?;
        let start =
            parse_process_identity(&read_kernel(&current, "stat", MAX_PROC_TEXT, true)?, pid)?
                .start_ticks;
        let original_membership =
            parse_membership(&read_kernel(&current, "cgroup", MAX_PROC_TEXT, true)?)?;
        let tasks = current
            .open_dir_nofollow("task")
            .map_err(|_| io("native_entry_tasks"))?;
        check(
            task_ids(&tasks, self.policy.pids_max)? == [pid],
            "native resource migration must precede all other threads",
        )?;
        let available =
            sched_getaffinity(Pid::from_raw(0)).map_err(|_| io("native_entry_affinity"))?;
        let mut expected = CpuSet::new();
        for &cpu in &self.policy.affinity {
            check(
                available
                    .is_set(cpu as usize)
                    .map_err(|_| invalid("native affinity CPU range"))?,
                "native requested affinity lies outside the current allowed CPUs",
            )?;
            expected
                .set(cpu as usize)
                .map_err(|_| invalid("native affinity CPU range"))?;
        }
        // Recheck the original single-thread identity immediately before the
        // first mutation. Future threads/processes inherit both boundaries.
        check(
            task_ids(&tasks, self.policy.pids_max)? == [pid]
                && parse_process_identity(
                    &read_kernel(&current, "stat", MAX_PROC_TEXT, true)?,
                    pid,
                )?
                .start_ticks
                    == start
                && parse_membership(&read_kernel(&current, "cgroup", MAX_PROC_TEXT, true)?)?
                    == original_membership,
            "native process identity changed before resource entry",
        )?;
        sched_setaffinity(Pid::from_raw(0), &expected)
            .map_err(|_| io("native_apply_self_affinity"))?;
        // This is deliberately outside write_kernel's unchanged limit-only
        // whitelist. The value comes only from the current process identity,
        // never from a public PID declaration or another process handle.
        write_cgroup_value(&leaf, "cgroup.procs", &std::process::id().to_string())?;
        let entry = RoventNativeProcessEntry {
            leaf: self.leaf_identity,
            root_pid: pid,
            root_start_ticks: start,
            original_membership,
            membership: self.membership.clone(),
        };
        self.observe_native_process_tree(&entry)?;
        Ok(entry)
    }

    /// Reuse the original held FD/readback without the trainer-specific
    /// startup label. This proves configured leaf controls only; the separate
    /// original entry and tree snapshot prove current membership/affinity.
    pub fn native_startup_json(&self) -> Result<Value, ArenaError> {
        let directory = self.recheck_identity()?;
        let readback = RoventTrainingResourceReadback::read(&directory)?;
        check(
            readback.matches(&self.policy),
            "native resource settings changed before observation",
        )?;
        Ok(serde_json::json!({
            "domain":"rz-rovent-native-resource-startup/1",
            "migration":"original_native_current_process_before_models_threads_or_children",
            "path":self.leaf_path,"device":self.leaf_identity.device,"inode":self.leaf_identity.inode,
            "membership":self.membership,"policy":self.policy,
            "prepared_readback":self.prepared_readback,"readback":readback,
            "available_controllers_snapshot":self.available_controllers,
            "subtree_controllers_snapshot":self.subtree_controllers,
            "scope":"configured_leaf_readback_not_complete_ancestry_membership_or_lifetime_proof"
        }))
    }

    pub fn observe_native_process_tree(
        &self,
        entry: &RoventNativeProcessEntry,
    ) -> Result<RoventNativeTreeObservation, ArenaError> {
        // A vanished task can race enumeration. Reobserve the *whole* tree,
        // never just accept the incomplete snapshot. Keep original evidence.
        let (mut observation, exited) = retry_exited_threads(|| self.observe_native_process_tree_attempt(entry))?;
        observation.exited_threads_reobserved = exited;
        Ok(observation)
    }

    fn observe_native_process_tree_attempt(
        &self,
        entry: &RoventNativeProcessEntry,
    ) -> Result<RoventNativeTreeObservation, NativeTreeAttemptError> {
        check(
            entry.leaf == self.leaf_identity
                && entry.membership == self.membership
                && entry.root_pid == std::process::id(),
            "foreign native resource entry",
        )?;
        let leaf = self.recheck_identity()?;
        let proc = proc_root()?;
        let before = native_tree(self, entry, &proc, &leaf)?;
        let readback = RoventTrainingResourceReadback::read(&leaf)?;
        check(
            readback.matches(&self.policy),
            "native leaf limits changed after entry",
        )?;
        let peak = read_number(&leaf, "memory.peak");
        let after = native_tree(self, entry, &proc, &leaf)?;
        check(
            before == after && RoventTrainingResourceReadback::read(&leaf)? == readback,
            "native process/thread membership changed during bounded observation",
        )?;
        self.recheck_identity()?;
        let kernel_threads = before
            .iter()
            .try_fold(0_u64, |count, process| {
                count.checked_add(process.thread_ids.len() as u64)
            })
            .ok_or_else(|| invalid("native tree thread count overflow"))?;
        Ok(RoventNativeTreeObservation {
            root_pid: entry.root_pid,
            root_start_ticks: entry.root_start_ticks,
            original_membership: entry.original_membership.clone(),
            membership: self.membership.clone(),
            processes: before,
            kernel_threads,
            readback,
            live_leaf_memory_peak_bytes: peak.as_ref().ok().copied(),
            live_leaf_memory_peak_unknown_reason: peak.err().map(|error| error.to_string()),
            exited_threads_reobserved: Vec::new(),
            scope: "bounded_current_native_descendant_and_thread_snapshot_not_lifetime_topology_ancestry_or_final_peak_proof",
        })
    }
}

#[cfg(test)]
mod reobservation_tests {
    use super::*;

    #[test]
    fn task_open_retries_only_verified_missing_thread() {
        let missing = || std::io::Error::from(std::io::ErrorKind::NotFound);
        assert!(matches!(task_open_failure(10, 11, missing(), || Ok(true)),
            NativeTreeAttemptError::ExitedThread(_)));
        assert!(matches!(task_open_failure(10, 11, missing(), || Ok(false)),
            NativeTreeAttemptError::Other(_)));
        assert!(matches!(task_open_failure(10, 11, missing(), || Err(ArenaError::Io("identity".into()))),
            NativeTreeAttemptError::Other(_)));
        let denied = std::io::Error::from(std::io::ErrorKind::PermissionDenied);
        let failure = task_open_failure(10, 11, denied, || panic!("permission is never retried"));
        let NativeTreeAttemptError::Other(error) = failure else { panic!("permission must fail"); };
        assert!(error.to_string().contains("pid=10 tid=11"));
        assert!(error.to_string().contains("PermissionDenied"));
    }

    #[test]
    fn only_confirmed_exited_thread_is_reobserved_with_original_evidence() {
        let mut calls = 0;
        let (value, exited) = retry_exited_threads(|| {
            calls += 1;
            if calls == 1 {
                Err(NativeTreeAttemptError::ExitedThread(RoventExitedThreadObservation {
                    pid: 10, tid: 11, errno: Some(2),
                }))
            } else { Ok(42) }
        }).unwrap();
        assert_eq!((value, calls, exited.len()), (42, 2, 1));
        assert_eq!((exited[0].pid, exited[0].tid, exited[0].errno), (10, 11, Some(2)));

        let mut failures = 0;
        let result = retry_exited_threads::<()>(|| {
            failures += 1;
            Err(NativeTreeAttemptError::Other(ArenaError::Io("permission".into())))
        });
        assert!(result.is_err());
        assert_eq!(failures, 1);

        let mut disappeared = 0;
        let result = retry_exited_threads::<()>(|| {
            disappeared += 1;
            Err(NativeTreeAttemptError::ExitedThread(RoventExitedThreadObservation {
                pid: 10, tid: 11, errno: Some(2),
            }))
        });
        assert!(result.is_err());
        assert_eq!(disappeared, 3);
    }
}

fn native_tree(
    resources: &RoventTrainingResources,
    entry: &RoventNativeProcessEntry,
    proc: &Dir,
    leaf: &Dir,
) -> Result<Vec<RoventNativeProcessSnapshot>, NativeTreeAttemptError> {
    let members = read_pids(leaf)?;
    check(
        members.contains(&entry.root_pid) && members.len() <= resources.policy.pids_max as usize,
        "native leaf membership capacity/root mismatch",
    )?;
    let mut expected = CpuSet::new();
    for &cpu in &resources.policy.affinity {
        expected
            .set(cpu as usize)
            .map_err(|_| invalid("native expected affinity"))?;
    }
    // Preserve the actual task/children edge rather than requiring every
    // descendant (or the root itself) to be a child of this caller.
    let mut pending = BTreeMap::from([(entry.root_pid, None)]);
    let mut visited = BTreeSet::new();
    let mut result = Vec::new();
    let mut thread_count = 0usize;
    while let Some((pid, parent)) = pending.pop_first() {
        check(
            members.contains(&pid) && visited.insert(pid),
            "native descendant escaped the leaf or formed a cycle",
        )?;
        let process = proc
            .open_dir_nofollow(pid.to_string())
            .map_err(|_| io("native_tree_process"))?;
        let identity =
            parse_process_identity(&read_kernel(&process, "stat", MAX_PROC_TEXT, true)?, pid)?;
        if let Some(parent) = parent {
            require_process_parent(&identity, parent)?;
        }
        let start_ticks = identity.start_ticks;
        check(
            pid != entry.root_pid || start_ticks == entry.root_start_ticks,
            "native original process PID was reused",
        )?;
        check(
            parse_membership(&read_kernel(&process, "cgroup", MAX_PROC_TEXT, true)?)?
                == entry.membership,
            "native process changed cgroup membership",
        )?;
        let tasks = process
            .open_dir_nofollow("task")
            .map_err(|_| io("native_tree_tasks"))?;
        let thread_ids = task_ids(&tasks, resources.policy.pids_max)?;
        thread_count = thread_count
            .checked_add(thread_ids.len())
            .ok_or_else(|| invalid("native thread capacity overflow"))?;
        check(
            thread_count <= resources.policy.pids_max as usize,
            "native process tree exceeds the original pids ceiling",
        )?;
        let mut children = BTreeSet::new();
        for &tid in &thread_ids {
            let task = match tasks.open_dir_nofollow(tid.to_string()) {
                Ok(task) => task,
                Err(error) => {
                    // Only ENOENT invokes this original-identity recheck.
                    return Err(task_open_failure(pid, tid, error, || {
                        let now = parse_process_identity(
                            &read_kernel(&process, "stat", MAX_PROC_TEXT, true)?, pid)?;
                        Ok(now.start_ticks == start_ticks
                            && !task_ids(&tasks, resources.policy.pids_max)?.contains(&tid))
                    }));
                }
            };
            check(
                parse_membership(&read_kernel(&task, "cgroup", MAX_PROC_TEXT, true)?)?
                    == entry.membership,
                "native thread changed cgroup membership",
            )?;
            let affinity = sched_getaffinity(Pid::from_raw(
                i32::try_from(tid).map_err(|_| invalid("native TID overflow"))?,
            ))
            .map_err(|_| io("native_tree_affinity"))?;
            check(
                affinity == expected,
                "native thread affinity differs from its inherited policy",
            )?;
            for child in read_kernel(&task, "children", MAX_PROC_TEXT, true)?.split_whitespace() {
                let child = canonical_u64(child)?;
                check(
                    child > 0 && child <= i32::MAX as u64,
                    "native descendant PID range",
                )?;
                children.insert(child as u32);
            }
        }
        let after =
            parse_process_identity(&read_kernel(&process, "stat", MAX_PROC_TEXT, true)?, pid)?;
        if let Some(parent) = parent {
            require_process_parent(&after, parent)?;
        }
        check(
            task_ids(&tasks, resources.policy.pids_max)? == thread_ids
                && after.start_ticks == start_ticks,
            "native process/thread identity changed during observation",
        )?;
        for &child in &children {
            check(
                !visited.contains(&child),
                "native descendant PID cycle/reuse",
            )?;
            check(
                pending.insert(child, Some(pid)).is_none(),
                "native descendant was listed under multiple parents",
            )?;
        }
        check(
            pending.len().saturating_add(visited.len()) <= resources.policy.pids_max as usize,
            "native descendants exceed original process inventory capacity",
        )?;
        result.push(RoventNativeProcessSnapshot {
            pid,
            start_ticks,
            thread_ids,
            children: children.into_iter().collect(),
            affinity: resources.policy.affinity.clone(),
        });
    }
    check(
        visited == members && read_pids(leaf)? == members,
        "native leaf has an unrelated process or changed during inventory",
    )?;
    result.sort_unstable_by_key(|process| process.pid);
    Ok(result)
}
