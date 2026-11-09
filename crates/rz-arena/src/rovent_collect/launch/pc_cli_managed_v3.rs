//! Syntax for the explicitly managed V3 Collection launch.
//!
//! Parsing file-descriptor numbers grants no authority. The authenticated
//! handoff and original supervisor must separately validate all four owners.
use super::{
    ArenaError, RoventPublicationFile, check, invalid, pc_lock_v3::PcCollectionCliV3,
    valid_relative,
};
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
    pub(crate) long_game_preflight: Option<RoventPublicationFile>,
    bootstrap_budget_ms: u64,
    pub(crate) work_deadline_ns: u64,
    pub(crate) cleanup_deadline_ns: u64,
}

impl ManagedCollectionCliV3 {
    /// Derive the local setup ceiling from the single process-entry sample.
    /// This budget is independent of native startup and parent work limits.
    pub(crate) fn bootstrap_deadline(&self, entry: Instant) -> Result<Instant, ArenaError> {
        entry
            .checked_add(Duration::from_millis(self.bootstrap_budget_ms))
            .ok_or_else(|| invalid("managed V3 bootstrap deadline overflow"))
    }

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
    parse_managed_collection_cli_for_v3(arguments, false)
}

/// Same managed transport, explicitly selecting the Own /3 plan domain.
pub(crate) fn parse_managed_own_collection_cli_v3(
    arguments: Vec<String>,
) -> Result<Option<ManagedCollectionCliV3>, ArenaError> {
    let selected = parse_managed_collection_cli_for_v3(arguments, true)?;
    check(
        selected
            .as_ref()
            .is_none_or(|value| value.long_game_preflight.is_none()),
        "Own V3 long-game preflight is not connected",
    )?;
    Ok(selected)
}

fn parse_managed_collection_cli_for_v3(
    arguments: Vec<String>,
    own: bool,
) -> Result<Option<ManagedCollectionCliV3>, ArenaError> {
    // Unrelated native V3 arguments do not select Collection. A partial
    // Collection selector still enters this parser and reports its own error.
    if !arguments.iter().any(|arg| {
        arg.starts_with("--v3-mode=")
            || arg.starts_with("--v3-plan=")
            || arg.starts_with("--v3-plan-sha256=")
            || arg.starts_with("--v3-plan-bytes=")
            || arg.starts_with("--v3-managed-")
            || arg.starts_with("--v3-bootstrap-budget-")
            || arg.starts_with("--v3-long-game-preflight-")
            || arg.starts_with("--v3-work-deadline-")
            || arg.starts_with("--v3-cleanup-deadline-")
    }) {
        return Ok(None);
    }
    let mut managed = BTreeMap::new();
    let mut selection_arguments = Vec::new();
    for argument in arguments {
        if argument.starts_with("--v3-managed-")
            || argument.starts_with("--v3-bootstrap-")
            || argument.starts_with("--v3-long-game-preflight-")
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
                        | "--v3-bootstrap-budget-ms"
                        | "--v3-long-game-preflight-path"
                        | "--v3-long-game-preflight-sha256"
                        | "--v3-long-game-preflight-bytes"
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
    let selection = if own {
        super::pc_lock_v3::parse_own_collection_cli_v3(selection_arguments)?
    } else {
        super::pc_lock_v3::parse_pc_collection_cli_v3(selection_arguments)?
    }
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
    let bootstrap_budget = take("--v3-bootstrap-budget-ms")?;
    check(
        bootstrap_budget.bytes().all(|byte| byte.is_ascii_digit()),
        "managed V3 bootstrap budget must be an integer",
    )?;
    let bootstrap_budget_ms: u64 = bootstrap_budget
        .parse()
        .map_err(|_| invalid("invalid managed V3 bootstrap budget"))?;
    check(
        bootstrap_budget_ms > 0,
        "managed V3 bootstrap budget must be positive",
    )?;
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
    let long_game_preflight = match (
        managed.remove("--v3-long-game-preflight-path"),
        managed.remove("--v3-long-game-preflight-sha256"),
        managed.remove("--v3-long-game-preflight-bytes"),
    ) {
        (None, None, None) => None,
        (Some(path), Some(sha256), Some(bytes)) => {
            let pin = RoventPublicationFile {
                path,
                sha256,
                bytes: bytes
                    .parse()
                    .map_err(|_| invalid("long-game preflight bytes"))?,
            };
            pin.validate()?;
            check(
                pin.path.starts_with("inputs/") && valid_relative(&pin.path),
                "long-game preflight input path",
            )?;
            Some(pin)
        }
        _ => {
            return Err(invalid(
                "long-game preflight requires path, sha256 and bytes",
            ));
        }
    };
    Ok(Some(ManagedCollectionCliV3 {
        selection,
        channel,
        output,
        input,
        plan,
        long_game_preflight,
        bootstrap_budget_ms,
        work_deadline_ns,
        cleanup_deadline_ns,
    }))
}

#[cfg(test)]
mod tests {
    use super::{parse_managed_collection_cli_v3, parse_managed_own_collection_cli_v3};

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
            "--v3-bootstrap-budget-ms=250".into(),
            "--native-argument=kept".into(),
        ]
    }

    #[test]
    fn unrelated_native_v3_arguments_do_not_select_collection() {
        assert!(
            parse_managed_collection_cli_v3(vec!["--v3-native-option=enabled".into()])
                .unwrap()
                .is_none()
        );
        assert!(parse_managed_collection_cli_v3(vec!["--v3-managed-plan-fd=3".into()]).is_err());
    }

    #[test]
    fn managed_selector_requires_distinct_fds_and_ordered_deadlines() {
        let selected = parse_managed_collection_cli_v3(arguments())
            .unwrap()
            .unwrap();
        assert!(selected.long_game_preflight.is_none());
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
        let entry = std::time::Instant::now();
        assert_eq!(
            selected.bootstrap_deadline(entry).unwrap(),
            entry + std::time::Duration::from_millis(250)
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
        let mut zero_budget = arguments();
        zero_budget[10] = "--v3-bootstrap-budget-ms=0".into();
        assert!(parse_managed_collection_cli_v3(zero_budget).is_err());
        let mut missing_budget = arguments();
        missing_budget.remove(10);
        assert!(parse_managed_collection_cli_v3(missing_budget).is_err());
    }

    #[test]
    fn own_managed_selection_keeps_transport_and_rejects_pc_crossing() {
        let mut own = arguments();
        own[0] = "--v3-mode=own".into();
        let selected = parse_managed_own_collection_cli_v3(own.clone())
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
        assert!(parse_managed_collection_cli_v3(own.clone()).is_err());
        own[7] = "--v3-managed-plan-fd=3".into();
        assert!(parse_managed_own_collection_cli_v3(own).is_err());
        assert!(parse_managed_own_collection_cli_v3(arguments()).is_err());
    }

    #[test]
    fn long_game_preflight_selector_requires_one_complete_pin() {
        let mut args = arguments();
        args.push("--v3-long-game-preflight-path=inputs/profile.json".into());
        assert!(parse_managed_collection_cli_v3(args.clone()).is_err());
        args.push(format!(
            "--v3-long-game-preflight-sha256={}",
            "b".repeat(64)
        ));
        args.push("--v3-long-game-preflight-bytes=1".into());
        let selected = parse_managed_collection_cli_v3(args.clone())
            .unwrap()
            .unwrap();
        assert_eq!(
            selected.long_game_preflight.unwrap().path,
            "inputs/profile.json"
        );
        args.push("--v3-long-game-preflight-bytes=1".into());
        assert!(parse_managed_collection_cli_v3(args).is_err());
    }
}
