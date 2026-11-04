//! Syntax for the explicitly managed V3 Collection launch.
//!
//! Parsing file-descriptor numbers grants no authority. The authenticated
//! handoff and original supervisor must separately validate all four owners.
use super::{ArenaError, check, invalid, pc_lock_v3::PcCollectionCliV3};
use std::collections::BTreeMap;
use std::os::fd::RawFd;
use std::time::{Duration, Instant};

#[derive(Debug)]
pub(crate) enum ManagedDeadlineErrorV3 {
    Namespace(super::managed_output_fd_v2::ManagedTimeNamespaceErrorV2),
    OriginalPlanDiffers,
    NamespaceOwnerDiffers,
    ClockUnavailable(nix::errno::Errno),
    ClockInvalid,
    WorkExpired,
    CleanupExpired,
    Overflow,
}

/// Raw proposed descriptors in the supervisor's /2 handoff order. They are
/// not opened, duplicated, closed, or treated as ownership by this parser.
pub(crate) struct ManagedCollectionCliV3 {
    pub(crate) selection: PcCollectionCliV3,
    pub(crate) channel: RawFd,
    pub(crate) output: RawFd,
    pub(crate) input: RawFd,
    pub(crate) plan: RawFd,
    pub(crate) work_deadline_ns: u64,
    pub(crate) cleanup_deadline_ns: u64,
}

impl ManagedCollectionCliV3 {
    /// Observe the same authenticated parent's and current thread's time
    /// namespace before interpreting its original monotonic ceilings. This
    /// is a point-in-time observation, not proof against a later setns or of
    /// where the parent originally created the supplied deadline. The caller
    /// must pass an independent local bootstrap ceiling for this observation.
    pub(crate) fn remaining_deadlines(
        &self,
        observation: &mut super::managed_output_fd_v2::ObservedManagedPlanLeaseV2,
        local_bootstrap_until: Instant,
    ) -> Result<(Instant, Instant), ManagedDeadlineErrorV3> {
        let pin = &self.selection.plan_pin;
        if !observation.matches_plan(&pin.path, pin.bytes, &pin.sha256) {
            return Err(ManagedDeadlineErrorV3::OriginalPlanDiffers);
        }
        let namespace = observation
            .observe_same_time_namespace(local_bootstrap_until)
            .map_err(ManagedDeadlineErrorV3::Namespace)?;
        if !namespace.matches_original(observation) {
            return Err(ManagedDeadlineErrorV3::NamespaceOwnerDiffers);
        }
        let entry = Instant::now();
        let now = nix::time::clock_gettime(nix::time::ClockId::CLOCK_MONOTONIC)
            .map_err(ManagedDeadlineErrorV3::ClockUnavailable)?;
        let seconds =
            u64::try_from(now.tv_sec()).map_err(|_| ManagedDeadlineErrorV3::ClockInvalid)?;
        let nanos =
            u64::try_from(now.tv_nsec()).map_err(|_| ManagedDeadlineErrorV3::ClockInvalid)?;
        let now_ns = seconds
            .checked_mul(1_000_000_000)
            .and_then(|value| value.checked_add(nanos))
            .ok_or(ManagedDeadlineErrorV3::Overflow)?;
        let work_ns = self
            .work_deadline_ns
            .checked_sub(now_ns)
            .filter(|remaining| *remaining > 0)
            .ok_or(ManagedDeadlineErrorV3::WorkExpired)?;
        let cleanup_ns = self
            .cleanup_deadline_ns
            .checked_sub(now_ns)
            .filter(|remaining| *remaining > 0)
            .ok_or(ManagedDeadlineErrorV3::CleanupExpired)?;
        let deadline = |remaining| {
            entry
                .checked_add(Duration::from_nanos(remaining))
                .ok_or(ManagedDeadlineErrorV3::Overflow)
        };
        Ok((deadline(work_ns)?, deadline(cleanup_ns)?))
    }
}

/// Extract only the managed transport options, then delegate the actual V3
/// selection to its existing parser. Unrelated native arguments retain order.
pub(crate) fn parse_managed_collection_cli_v3(
    arguments: Vec<String>,
) -> Result<Option<ManagedCollectionCliV3>, ArenaError> {
    if !arguments.iter().any(|arg| arg.starts_with("--v3-")) {
        return Ok(None);
    }
    let mut managed = BTreeMap::new();
    let mut selection_arguments = Vec::new();
    for argument in arguments {
        if argument.starts_with("--v3-managed-")
            || argument.starts_with("--v3-work-deadline-")
            || argument.starts_with("--v3-cleanup-deadline-")
        {
            let (key, value) = argument
                .split_once('=')
                .ok_or_else(|| invalid("managed V3 options require =value"))?;
            check(
                matches!(
                    key,
                    "--v3-managed-output-channel-fd"
                        | "--v3-managed-output-dir-fd"
                        | "--v3-managed-input-dir-fd"
                        | "--v3-managed-plan-fd"
                        | "--v3-work-deadline-monotonic-ns"
                        | "--v3-cleanup-deadline-monotonic-ns"
                ) && !value.is_empty()
                    && managed.insert(key.to_owned(), value.to_owned()).is_none(),
                "unknown, empty or duplicate managed V3 option",
            )?;
        } else {
            selection_arguments.push(argument);
        }
    }
    let selection = super::pc_lock_v3::parse_pc_collection_cli_v3(selection_arguments)?
        .ok_or_else(|| invalid("missing V3 Collection selector"))?;
    let mut take = |key: &str| {
        managed
            .remove(key)
            .ok_or_else(|| invalid(format!("missing {key}")))
    };
    let mut fd = |key: &str| -> Result<RawFd, ArenaError> {
        let raw: RawFd = take(key)?
            .parse()
            .map_err(|_| invalid("managed V3 fd must be an integer"))?;
        check(raw >= 0, "managed V3 fd must be nonnegative")?;
        Ok(raw)
    };
    let channel = fd("--v3-managed-output-channel-fd")?;
    let output = fd("--v3-managed-output-dir-fd")?;
    let input = fd("--v3-managed-input-dir-fd")?;
    let plan = fd("--v3-managed-plan-fd")?;
    check(
        [channel, output, input, plan]
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            == 4,
        "managed V3 descriptors must be distinct",
    )?;
    // Native owns its observer descriptor independently. Reusing the same
    // number would give two consumers ownership of one inherited descriptor.
    for argument in &selection.native_arguments {
        if let Some(value) = argument.strip_prefix("--go-observer-fd=") {
            if let Ok(observer) = value.parse::<RawFd>() {
                check(
                    ![channel, output, input, plan].contains(&observer),
                    "managed V3 descriptor conflicts with native observer",
                )?;
            }
        }
    }
    let work_deadline_ns: u64 = take("--v3-work-deadline-monotonic-ns")?
        .parse()
        .map_err(|_| invalid("invalid managed V3 work deadline"))?;
    let cleanup_deadline_ns: u64 = take("--v3-cleanup-deadline-monotonic-ns")?
        .parse()
        .map_err(|_| invalid("invalid managed V3 cleanup deadline"))?;
    check(
        work_deadline_ns > 0 && work_deadline_ns <= cleanup_deadline_ns,
        "managed V3 deadlines must be positive and ordered",
    )?;
    Ok(Some(ManagedCollectionCliV3 {
        selection,
        channel,
        output,
        input,
        plan,
        work_deadline_ns,
        cleanup_deadline_ns,
    }))
}

#[cfg(test)]
mod tests {
    use super::parse_managed_collection_cli_v3;

    fn arguments() -> Vec<String> {
        vec![
            "--v3-mode=pc".into(),
            "--v3-plan=inputs/plan.json".into(),
            format!("--v3-plan-sha256={}", "a".repeat(64)),
            "--v3-plan-bytes=1".into(),
            "--v3-managed-output-channel-fd=3".into(),
            "--v3-managed-output-dir-fd=4".into(),
            "--v3-managed-input-dir-fd=5".into(),
            "--v3-managed-plan-fd=6".into(),
            "--v3-work-deadline-monotonic-ns=100".into(),
            "--v3-cleanup-deadline-monotonic-ns=120".into(),
            "--native-argument=kept".into(),
        ]
    }

    #[test]
    fn managed_selector_requires_distinct_fds_and_ordered_deadlines() {
        let selected = parse_managed_collection_cli_v3(arguments())
            .unwrap()
            .unwrap();
        assert_eq!(
            [
                selected.channel,
                selected.output,
                selected.input,
                selected.plan
            ],
            [3, 4, 5, 6]
        );
        assert_eq!(
            selected.selection.native_arguments,
            ["--native-argument=kept"]
        );
        let mut duplicate_fd = arguments();
        duplicate_fd[7] = "--v3-managed-plan-fd=3".into();
        assert!(parse_managed_collection_cli_v3(duplicate_fd).is_err());
        let mut missing_fd = arguments();
        missing_fd.remove(7);
        assert!(parse_managed_collection_cli_v3(missing_fd).is_err());
        let mut reversed = arguments();
        reversed[9] = "--v3-cleanup-deadline-monotonic-ns=99".into();
        assert!(parse_managed_collection_cli_v3(reversed).is_err());
        let mut mixed = arguments();
        mixed.push("--v2-mode=pc".into());
        assert!(parse_managed_collection_cli_v3(mixed).is_err());
        let mut observer_collision = arguments();
        observer_collision.push("--go-observer-fd=3".into());
        assert!(parse_managed_collection_cli_v3(observer_collision).is_err());
    }
}
