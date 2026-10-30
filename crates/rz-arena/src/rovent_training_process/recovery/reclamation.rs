//! Admission of an old run's retirement scope. A completed pinned historical
//! closure may be consumed on a different host. An unresolved run requires a
//! fresh same-boot cgroup/PID observation; paths and JSON claims alone cannot
//! issue that owner. No method signals, waits for, or reaps a numeric old PID.
use super::*;
use cap_fs_ext::{DirExt, FollowSymlinks, OpenOptionsFollowExt};
use cap_std::{ambient_authority, fs::Dir};
use nix::sys::statfs::{CGROUP2_SUPER_MAGIC, PROC_SUPER_MAGIC, fstatfs};
use std::{io::Read, os::unix::fs::MetadataExt};

struct LocalOldScope {
    identity: RoventHistoricalChildIdentity,
    directory: Dir,
}

/// Current OS observations for explicitly unresolved historical scopes only.
/// Callers must classify fully pinned completed closures before choosing this
/// list; a completed closure does not need its old host or cgroup to survive.
/// This owner grants no numeric PID signal/reap or old physical capability.
pub(in crate::rovent_training_process) struct VerifiedHistoricalUnresolvedScopes {
    scopes: Vec<LocalOldScope>,
    budget: Arc<RoventRecoveryReadBudget>,
}
impl VerifiedHistoricalUnresolvedScopes {
    pub(in crate::rovent_training_process) fn admit(
        identities: &[RoventHistoricalChildIdentity],
        budget: Arc<RoventRecoveryReadBudget>,
    ) -> Result<Self, ArenaError> {
        check(
            identities.len() <= 64,
            "bounded unresolved historical scope list",
        )?;
        budget.reserve(0)?;
        let mut seen = BTreeSet::new();
        let mut scopes = Vec::with_capacity(identities.len());
        for identity in identities {
            check(
                seen.insert((
                    identity.boot_id.as_str(),
                    identity.pid,
                    identity.start_ticks,
                    identity.cgroup_path.as_str(),
                    identity.cgroup_device,
                    identity.cgroup_inode,
                )),
                "duplicate original unresolved scope identity",
            )?;
            let scope = LocalOldScope {
                identity: identity.clone(),
                directory: open_absolute(Path::new(&identity.cgroup_path))?,
            };
            observe_old_scope(&scope, &budget)?;
            scopes.push(scope);
        }
        Ok(Self { scopes, budget })
    }

    pub(in crate::rovent_training_process) fn revalidate(&self) -> Result<(), ArenaError> {
        self.budget.reserve(0)?;
        for scope in &self.scopes {
            observe_old_scope(scope, &self.budget)?;
        }
        Ok(())
    }
}
pub struct VerifiedRoventOldRunReclamation {
    checkpoint: Arc<VerifiedRoventCheckpointSource>,
    original_external_closure: HeldHistoricalFile,
    local_scopes: Vec<LocalOldScope>,
    historical_closed: usize,
    budget: Arc<RoventRecoveryReadBudget>,
}
impl VerifiedRoventOldRunReclamation {
    pub fn admit(
        checkpoint: Arc<VerifiedRoventCheckpointSource>,
        requested_closure: &RoventTrainingFilePin,
        budget: Arc<RoventRecoveryReadBudget>,
    ) -> Result<Self, ArenaError> {
        let declared = &checkpoint.publication().recorded_closure.evidence;
        check(
            requested_closure.sha256 == declared.sha256
                && requested_closure.bytes == declared.bytes,
            "reclamation request does not pin the selected checkpoint's original closure",
        )?;
        let (original_external_closure, _) =
            budget.read(requested_closure, MAX_METADATA_BYTES, false)?;
        let mut local_scopes = Vec::new();
        let mut historical_closed = 0;
        let mut source = Some(&checkpoint);
        let mut windows = BTreeSet::new();
        while let Some(owner) = source {
            let (closure, classification) = owner.recorded_closure();
            check(
                windows.insert((closure.execution_id.clone(), closure.window_id.clone())),
                "old-run closure chain repeats a window",
            )?;
            match classification {
                RoventHistoricalClosureClassification::RecordedClosed => historical_closed += 1,
                RoventHistoricalClosureClassification::Unresolved => {
                    // Different/missing host evidence is not an implicit kill.
                    let directory = open_absolute(Path::new(&closure.child_identity.cgroup_path))?;
                    let scope = LocalOldScope {
                        identity: closure.child_identity.clone(),
                        directory,
                    };
                    observe_old_scope(&scope, &budget)?;
                    local_scopes.push(scope);
                }
            }
            source = owner.parent();
        }
        Ok(Self {
            checkpoint,
            original_external_closure,
            local_scopes,
            historical_closed,
            budget,
        })
    }
    pub fn checkpoint_publication_sha256(&self) -> &str {
        &self.checkpoint.publication_pin().sha256
    }
    pub fn observation(&self) -> Value {
        json!({"schema":"rz-rovent-old-run-reclamation-observation/1",
            "checkpoint_publication_sha256":self.checkpoint_publication_sha256(),
            "pinned_completed_historical_windows":self.historical_closed,
            "currently_observed_local_unresolved_scopes":self.local_scopes.len(),
            "past_native_authority_reissued":false,"past_physical_completion_reissued":false,
            "current_host_gpu_exclusivity":"unknown",
            "scope":"recorded_closed_or_same_boot_original_cgroup_empty_and_old_pid_absent_at_admission"})
    }
    pub fn revalidate(&self) -> Result<(), ArenaError> {
        self.budget.verify_held(
            &self.original_external_closure.file,
            &self.original_external_closure.pin,
            false,
        )?;
        for scope in &self.local_scopes {
            observe_old_scope(scope, &self.budget)?;
        }
        Ok(())
    }
}
fn observe_old_scope(
    scope: &LocalOldScope,
    budget: &RoventRecoveryReadBudget,
) -> Result<(), ArenaError> {
    let proc = open_absolute(Path::new("/proc"))?;
    ensure_fs(&directory_file(&proc)?, true)?;
    let boot = proc
        .open_dir_nofollow("sys")
        .and_then(|d| d.open_dir_nofollow("kernel"))
        .and_then(|d| d.open_dir_nofollow("random"))
        .map_err(|_| invalid("old-run boot scope unavailable"))?;
    check(
        kernel_read(&boot, "boot_id", 128, true, budget)?.trim() == scope.identity.boot_id,
        "unresolved prior run is on a different or unobserved boot; external reclamation evidence is required",
    )?;
    let current = open_absolute(Path::new(&scope.identity.cgroup_path))?;
    for directory in [&scope.directory, &current] {
        let file = directory_file(directory)?;
        ensure_fs(&file, false)?;
        let stat = file
            .metadata()
            .map_err(|_| invalid("old cgroup metadata unavailable"))?;
        check(
            stat.dev() == scope.identity.cgroup_device && stat.ino() == scope.identity.cgroup_inode,
            "old-run cgroup identity was removed or replaced",
        )?;
        check(
            kernel_read(directory, "cgroup.procs", 64 << 10, false, budget)?
                .trim()
                .is_empty(),
            "unresolved old-run cgroup still contains direct processes",
        )?;
        let events = kernel_read(directory, "cgroup.events", 4096, false, budget)?;
        let populated = events
            .lines()
            .filter_map(|line| line.strip_prefix("populated "))
            .collect::<Vec<_>>();
        check(
            populated == ["0"],
            "unresolved old-run cgroup or descendants remain populated/unknown",
        )?;
    }
    match proc.open_dir_nofollow(scope.identity.pid.to_string()) {
        Ok(pid) => {
            let stat = kernel_read(&pid, "stat", 16 << 10, true, budget)?;
            let rest = stat
                .rsplit_once(") ")
                .map(|(_, rest)| rest)
                .ok_or_else(|| invalid("old PID stat framing"))?;
            let ticks = rest
                .split_whitespace()
                .nth(19)
                .and_then(|value| value.parse::<u64>().ok())
                .ok_or_else(|| invalid("old PID start identity unavailable"))?;
            check(
                ticks != scope.identity.start_ticks,
                "original unresolved old process is still alive",
            )?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => (),
        Err(_) => return Err(invalid("old PID liveness is unknown")),
    }
    Ok(())
}
pub(super) fn open_recovery_directory(path: &Path) -> Result<File, ArenaError> {
    directory_file(&open_absolute(path)?)
}
fn open_absolute(path: &Path) -> Result<Dir, ArenaError> {
    check(
        path.is_absolute()
            && path
                .components()
                .all(|part| matches!(part, Component::RootDir | Component::Normal(_))),
        "old-run observation needs a normalized absolute path",
    )?;
    let mut directory = Dir::open_ambient_dir("/", ambient_authority())
        .map_err(|_| invalid("old-run root open"))?;
    for part in path.components() {
        if let Component::Normal(name) = part {
            directory = directory
                .open_dir_nofollow(name)
                .map_err(|_| invalid("old-run nofollow directory unavailable"))?;
        }
    }
    Ok(directory)
}
fn directory_file(directory: &Dir) -> Result<File, ArenaError> {
    let mut options = cap_std::fs::OpenOptions::new();
    options.read(true).follow(FollowSymlinks::No);
    directory
        .open_with(".", &options)
        .map(|file| file.into_std())
        .map_err(|_| invalid("old-run directory descriptor"))
}
fn ensure_fs(file: &File, procfs: bool) -> Result<(), ArenaError> {
    let expected = if procfs {
        PROC_SUPER_MAGIC
    } else {
        CGROUP2_SUPER_MAGIC
    };
    check(
        fstatfs(file)
            .map_err(|_| invalid("old-run filesystem readback"))?
            .filesystem_type()
            == expected,
        "old-run observation is not the registered kernel filesystem",
    )
}
fn kernel_read(
    directory: &Dir,
    name: &str,
    maximum: usize,
    procfs: bool,
    budget: &RoventRecoveryReadBudget,
) -> Result<String, ArenaError> {
    check(
        name.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._".contains(&b)),
        "fixed kernel basename required",
    )?;
    budget.reserve(maximum as u64)?;
    let mut options = cap_std::fs::OpenOptions::new();
    options.read(true).follow(FollowSymlinks::No);
    let mut file = directory
        .open_with(name, &options)
        .map(|file| file.into_std())
        .map_err(|_| invalid("old-run kernel read unavailable"))?;
    ensure_fs(&file, procfs)?;
    let mut raw = Vec::new();
    (&mut file)
        .take(maximum as u64 + 1)
        .read_to_end(&mut raw)
        .map_err(|_| invalid("old-run kernel read failed"))?;
    check(
        raw.len() <= maximum,
        "old-run kernel read exceeded its finite bound",
    )?;
    String::from_utf8(raw).map_err(|_| invalid("old-run kernel observation is not UTF8"))
}
