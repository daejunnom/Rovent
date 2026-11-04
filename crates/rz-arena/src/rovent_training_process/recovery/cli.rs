//! Product CLI request parsing for the recovered trainer.
//!
//! This file deliberately does not open files, create a cgroup, import Python,
//! start a process, or manufacture a native/physical completion capability.
//! The thin product binary passes this request to the concrete recovered-training
//! owner after registering its own cancellation hooks. The owner's admission
//! must compare every requested bound with the exact pinned registration.

use super::super::{MAX_TRAINER_OUTPUT_BYTES, RoventTrainingFilePin, invalid};
use crate::ArenaError;
use rz_experiments::pals_data_v3::{generation_identifier, generation_sha};
use std::{
    collections::BTreeMap,
    ffi::OsString,
    path::{Component, Path, PathBuf},
    time::{Duration, Instant},
};

pub const ROVENT_TRAINER_CLI_USAGE: &str = "rz-rovent-trainer --action <inspect|execute> --registration <absolute.json> --registration-sha256 <sha256> --registration-bytes <bytes> --wall-ms <ms> --cleanup-ms <ms> --read-bytes <bytes> --output-bytes <bytes> --stdio-bytes <bytes> --cpu-millicpus <quota> --memory-high-bytes <bytes> --memory-max-bytes <bytes> --pids-max <count> --affinity <sorted-cpu-list> --device <cpu|cuda:N> --delegated-cgroup <absolute-directory> --cgroup-leaf <new-leaf> --total-quota <strict|conditional>";

const MAX_REGISTRATION_BYTES: u64 = 16 << 20;
const MAX_STDIO_BYTES: u64 = 128 << 10;
const MAX_PATH_BYTES: usize = 4096;
const OPTIONS: [&str; 18] = [
    "--action",
    "--registration",
    "--registration-sha256",
    "--registration-bytes",
    "--wall-ms",
    "--cleanup-ms",
    "--read-bytes",
    "--output-bytes",
    "--stdio-bytes",
    "--cpu-millicpus",
    "--memory-high-bytes",
    "--memory-max-bytes",
    "--pids-max",
    "--affinity",
    "--device",
    "--delegated-cgroup",
    "--cgroup-leaf",
    "--total-quota",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoventTrainerCliAction {
    /// Exact historical-data inspection only; never execution admission.
    Inspect,
    /// Ask the concrete owner to admit and run the registration's own action.
    /// The pinned registration still distinguishes train from export.
    Execute,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoventTrainerCliTotalQuota {
    /// The owner must observe an enforced parent-plus-child aggregate scope.
    Strict,
    /// Unknown aggregate limits are reportable, never performance evidence.
    Conditional,
}

/// Requested bounds, not evidence that Linux applied any of them. In
/// particular, a path and leaf name cannot stand in for an owned cgroup FD.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RoventTrainerCliScopeRequest {
    pub wall_ms: u64,
    pub cleanup_ms: u64,
    pub max_read_bytes: u64,
    pub max_output_bytes: u64,
    pub max_stdio_bytes: u64,
    pub cpu_quota_millicpus: u32,
    pub memory_high_bytes: u64,
    pub memory_max_bytes: u64,
    pub pids_max: u32,
    pub affinity: Vec<u32>,
    pub device: String,
    pub delegated_cgroup: PathBuf,
    pub cgroup_leaf: String,
    pub total_quota: RoventTrainerCliTotalQuota,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RoventTrainerCliCommand {
    pub action: RoventTrainerCliAction,
    pub registration: RoventTrainingFilePin,
    pub scope: RoventTrainerCliScopeRequest,
}

#[derive(Clone, Debug, Eq, PartialEq)]
// Parsing returns one command-or-help decision. Keep the public by-value
// command API rather than changing its ownership contract for enum sizing.
#[allow(clippy::large_enum_variant)]
pub enum RoventTrainerCliParse {
    Help,
    Command(RoventTrainerCliCommand),
}

/// Deadlines anchored once to the thin binary's original entry time. This is
/// still an unadmitted request, not a launch/window/native completion witness.
/// No retry helper exists that silently obtains a fresh time allowance.
pub struct RoventTrainerCliAdmissionRequest {
    command: RoventTrainerCliCommand,
    original_started: Instant,
    execution_deadline: Instant,
    whole_deadline: Instant,
}

impl RoventTrainerCliCommand {
    pub fn anchor_original_window(
        self,
        original_started: Instant,
    ) -> Result<RoventTrainerCliAdmissionRequest, ArenaError> {
        self.validate()?;
        let execution_deadline = original_started
            .checked_add(Duration::from_millis(self.scope.wall_ms))
            .ok_or_else(|| invalid("CLI execution deadline is not representable"))?;
        let whole_deadline = execution_deadline
            .checked_add(Duration::from_millis(self.scope.cleanup_ms))
            .ok_or_else(|| invalid("CLI cleanup deadline is not representable"))?;
        let now = Instant::now();
        require(
            original_started <= now && now < execution_deadline,
            "CLI original execution window already expired or begins in the future",
        )?;
        Ok(RoventTrainerCliAdmissionRequest {
            command: self,
            original_started,
            execution_deadline,
            whole_deadline,
        })
    }

    pub fn validate(&self) -> Result<(), ArenaError> {
        let scope = &self.scope;
        require(
            exact_absolute_path(&self.registration.path)
                && self
                    .registration
                    .path
                    .extension()
                    .is_some_and(|e| e == "json")
                && generation_sha(&self.registration.sha256)
                && (1..=MAX_REGISTRATION_BYTES).contains(&self.registration.bytes),
            "CLI requires an exact absolute registration JSON pin",
        )?;
        require(
            scope.wall_ms > 0
                && scope.cleanup_ms > 0
                && scope.wall_ms.checked_add(scope.cleanup_ms).is_some()
                && scope.max_read_bytes >= self.registration.bytes
                && scope.max_read_bytes <= MAX_TRAINER_OUTPUT_BYTES * 260
                && (1..=MAX_TRAINER_OUTPUT_BYTES).contains(&scope.max_output_bytes)
                && (1..=MAX_STDIO_BYTES).contains(&scope.max_stdio_bytes),
            "CLI requires finite nonzero time/read/artifact/stdio bounds",
        )?;
        require(
            (1..=256_000).contains(&scope.cpu_quota_millicpus)
                && scope.memory_high_bytes > 0
                && scope.memory_high_bytes <= scope.memory_max_bytes
                && scope.memory_max_bytes <= 1_u64 << 50
                && (1..=4096).contains(&scope.pids_max)
                && !scope.affinity.is_empty()
                && scope.affinity.len() <= 1024
                && scope.affinity.iter().all(|cpu| *cpu < 1024)
                && scope.affinity.windows(2).all(|pair| pair[0] < pair[1])
                && valid_device(&scope.device),
            "CLI CPU/memory/pids/affinity/device request is not bounded and canonical",
        )?;
        require(
            exact_absolute_path(&scope.delegated_cgroup)
                && generation_identifier(&scope.cgroup_leaf)
                && scope.cgroup_leaf.len() <= 128
                && !scope.cgroup_leaf.contains(['/', '\\', ':'])
                && Path::new(&scope.cgroup_leaf).components().count() == 1
                && Path::new(&scope.cgroup_leaf)
                    .components()
                    .all(|part| matches!(part, Component::Normal(_))),
            "CLI requires an absolute delegated scope and a single new leaf name",
        )
    }
}

impl RoventTrainerCliAdmissionRequest {
    pub fn command(&self) -> &RoventTrainerCliCommand {
        &self.command
    }
    pub fn original_started(&self) -> Instant {
        self.original_started
    }
    pub fn execution_deadline(&self) -> Instant {
        self.execution_deadline
    }
    pub fn whole_deadline(&self) -> Instant {
        self.whole_deadline
    }
    pub fn ensure_admission_open(&self) -> Result<(), ArenaError> {
        require(
            Instant::now() < self.execution_deadline,
            "CLI original admission window expired; no replacement window was issued",
        )
    }
}

/// Parse `args_os().skip(1)`. All options are explicit, occur once, use separate
/// values, and have no environment/default fallback. Unknown input is rejected
/// without copying raw arguments, paths or possible credentials into errors.
pub fn parse_rovent_trainer_cli(
    arguments: impl IntoIterator<Item = OsString>,
) -> Result<RoventTrainerCliParse, ArenaError> {
    let arguments: Vec<OsString> = arguments.into_iter().take(OPTIONS.len() * 2 + 1).collect();
    if arguments.len() == 1 && arguments[0] == "--help" {
        return Ok(RoventTrainerCliParse::Help);
    }
    require(
        arguments.len() == OPTIONS.len() * 2,
        "CLI requires each documented option exactly once; use --help",
    )?;
    let mut values = BTreeMap::new();
    for pair in arguments.chunks_exact(2) {
        let option = pair[0]
            .to_str()
            .ok_or_else(|| invalid("CLI option is not UTF8"))?;
        let value = pair[1]
            .to_str()
            .filter(|value| !value.is_empty() && value.len() <= MAX_PATH_BYTES)
            .ok_or_else(|| invalid("CLI value is empty, non-UTF8 or exceeds its bound"))?;
        require(OPTIONS.contains(&option), "CLI option is unknown")?;
        require(
            values.insert(option, value).is_none(),
            "CLI option was supplied more than once",
        )?;
    }
    let value = |key| {
        values
            .get(key)
            .copied()
            .ok_or_else(|| invalid("CLI required option is missing"))
    };
    let action = match value("--action")? {
        "inspect" => RoventTrainerCliAction::Inspect,
        "execute" => RoventTrainerCliAction::Execute,
        _ => return Err(invalid("CLI action must be inspect or execute")),
    };
    let total_quota = match value("--total-quota")? {
        "strict" => RoventTrainerCliTotalQuota::Strict,
        "conditional" => RoventTrainerCliTotalQuota::Conditional,
        _ => return Err(invalid("CLI total quota must be strict or conditional")),
    };
    let number = |key| canonical_u64(value(key)?);
    let small_number =
        |key| u32::try_from(number(key)?).map_err(|_| invalid("CLI integer exceeds u32"));
    let affinity = value("--affinity")?
        .split(',')
        .map(|cpu| {
            u32::try_from(canonical_u64(cpu)?)
                .map_err(|_| invalid("CLI affinity index exceeds u32"))
        })
        .collect::<Result<Vec<_>, ArenaError>>()?;
    let command = RoventTrainerCliCommand {
        action,
        registration: RoventTrainingFilePin {
            path: PathBuf::from(value("--registration")?),
            sha256: value("--registration-sha256")?.into(),
            bytes: number("--registration-bytes")?,
        },
        scope: RoventTrainerCliScopeRequest {
            wall_ms: number("--wall-ms")?,
            cleanup_ms: number("--cleanup-ms")?,
            max_read_bytes: number("--read-bytes")?,
            max_output_bytes: number("--output-bytes")?,
            max_stdio_bytes: number("--stdio-bytes")?,
            cpu_quota_millicpus: small_number("--cpu-millicpus")?,
            memory_high_bytes: number("--memory-high-bytes")?,
            memory_max_bytes: number("--memory-max-bytes")?,
            pids_max: small_number("--pids-max")?,
            affinity,
            device: value("--device")?.into(),
            delegated_cgroup: PathBuf::from(value("--delegated-cgroup")?),
            cgroup_leaf: value("--cgroup-leaf")?.into(),
            total_quota,
        },
    };
    command.validate()?;
    Ok(RoventTrainerCliParse::Command(command))
}

fn require(condition: bool, reason: &str) -> Result<(), ArenaError> {
    if condition {
        Ok(())
    } else {
        Err(invalid(reason))
    }
}
fn canonical_u64(value: &str) -> Result<u64, ArenaError> {
    require(
        !value.is_empty()
            && value.len() <= 20
            && value.bytes().all(|byte| byte.is_ascii_digit())
            && (value == "0" || !value.starts_with('0')),
        "CLI integer must use canonical unsigned decimal notation",
    )?;
    value
        .parse()
        .map_err(|_| invalid("CLI integer exceeds u64"))
}
fn exact_absolute_path(path: &Path) -> bool {
    path.is_absolute()
        && path.as_os_str().len() <= MAX_PATH_BYTES
        && path
            .components()
            .all(|part| !matches!(part, Component::CurDir | Component::ParentDir))
        && path
            .to_str()
            .is_some_and(|text| !text.chars().any(char::is_control) && !text.contains('\0'))
}
fn valid_device(value: &str) -> bool {
    value == "cpu"
        || value
            .strip_prefix("cuda:")
            .is_some_and(|index| canonical_u64(index).is_ok_and(|number| number < 64))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arguments() -> Vec<OsString> {
        let root = if cfg!(windows) {
            "C:/rovent"
        } else {
            "/rovent"
        };
        [
            ("--action", "inspect".to_owned()),
            ("--registration", format!("{root}/registration.json")),
            ("--registration-sha256", "1".repeat(64)),
            ("--registration-bytes", "512".to_owned()),
            ("--wall-ms", "60000".to_owned()),
            ("--cleanup-ms", "30000".to_owned()),
            ("--read-bytes", "4096".to_owned()),
            ("--output-bytes", "1048576".to_owned()),
            ("--stdio-bytes", "65536".to_owned()),
            ("--cpu-millicpus", "2000".to_owned()),
            ("--memory-high-bytes", "6442450944".to_owned()),
            ("--memory-max-bytes", "12884901888".to_owned()),
            ("--pids-max", "64".to_owned()),
            ("--affinity", "0,2".to_owned()),
            ("--device", "cpu".to_owned()),
            ("--delegated-cgroup", format!("{root}/delegated")),
            ("--cgroup-leaf", "recovered-window-1".to_owned()),
            ("--total-quota", "strict".to_owned()),
        ]
        .into_iter()
        .flat_map(|(key, value)| [OsString::from(key), OsString::from(value)])
        .collect()
    }

    fn replace(arguments: &mut [OsString], key: &str, value: &str) {
        let slot = arguments.iter().position(|item| item == key).unwrap();
        arguments[slot + 1] = value.into();
    }

    #[test]
    fn only_explicit_standalone_help_skips_the_request() {
        assert!(matches!(
            parse_rovent_trainer_cli([OsString::from("--help")]).unwrap(),
            RoventTrainerCliParse::Help
        ));
        assert!(parse_rovent_trainer_cli(Vec::<OsString>::new()).is_err());
        assert!(parse_rovent_trainer_cli(["--help".into(), "execute".into()]).is_err());
    }

    #[test]
    fn inspect_and_execute_remain_unadmitted_explicit_requests() {
        let mut arguments = arguments();
        for action in ["inspect", "execute"] {
            replace(&mut arguments, "--action", action);
            let RoventTrainerCliParse::Command(command) =
                parse_rovent_trainer_cli(arguments.clone()).unwrap()
            else {
                panic!("explicit command required")
            };
            assert_eq!(command.scope.affinity, [0, 2]);
            assert_eq!(command.registration.bytes, 512);
            assert_eq!(
                command.action,
                if action == "inspect" {
                    RoventTrainerCliAction::Inspect
                } else {
                    RoventTrainerCliAction::Execute
                }
            );
        }
    }

    #[test]
    fn duplicate_unknown_missing_and_extra_options_are_rejected() {
        let mut duplicate = arguments();
        duplicate[2] = "--action".into();
        assert!(parse_rovent_trainer_cli(duplicate).is_err());
        let mut unknown = arguments();
        unknown[0] = "--unknown".into();
        assert!(parse_rovent_trainer_cli(unknown).is_err());
        let mut missing = arguments();
        missing.truncate(missing.len() - 2);
        assert!(parse_rovent_trainer_cli(missing).is_err());
        let mut extra = arguments();
        extra.extend(["--action".into(), "execute".into()]);
        assert!(parse_rovent_trainer_cli(extra).is_err());
    }

    #[test]
    fn malformed_or_inconsistent_bounds_never_reach_admission() {
        for (key, value) in [
            ("--wall-ms", "0"),
            ("--cleanup-ms", "0"),
            ("--wall-ms", "060000"),
            ("--read-bytes", "511"),
            ("--output-bytes", "0"),
            ("--stdio-bytes", "131073"),
            ("--cpu-millicpus", "+2000"),
            ("--pids-max", "4294967296"),
            ("--memory-high-bytes", "12884901889"),
            ("--registration-sha256", "unknown"),
            ("--registration-bytes", "16777217"),
            ("--device", "cuda:01"),
            ("--device", "cuda:64"),
            ("--affinity", "2,0"),
            ("--affinity", "0,0"),
            ("--total-quota", "observed"),
        ] {
            let mut input = arguments();
            replace(&mut input, key, value);
            assert!(parse_rovent_trainer_cli(input).is_err(), "{key}");
        }
    }

    #[test]
    fn scope_and_registration_must_be_absolute_single_purpose_paths() {
        for (key, value) in [
            ("--registration", "registration.json"),
            ("--registration", "/rovent/../registration.json"),
            ("--registration", "/rovent/input.bin"),
            ("--delegated-cgroup", "relative/delegated"),
            ("--cgroup-leaf", "first/second"),
            ("--cgroup-leaf", "first\\second"),
            ("--cgroup-leaf", ".."),
            ("--cgroup-leaf", "leaf:stream"),
        ] {
            let mut input = arguments();
            replace(&mut input, key, value);
            assert!(parse_rovent_trainer_cli(input).is_err(), "{key}");
        }
    }

    #[test]
    fn original_deadlines_do_not_refresh_an_expired_window() {
        let RoventTrainerCliParse::Command(command) =
            parse_rovent_trainer_cli(arguments()).unwrap()
        else {
            panic!("explicit command required")
        };
        let started = Instant::now();
        let request = command.clone().anchor_original_window(started).unwrap();
        assert_eq!(request.original_started(), started);
        assert_eq!(
            request.execution_deadline().duration_since(started),
            Duration::from_millis(60_000)
        );
        assert_eq!(
            request.whole_deadline().duration_since(started),
            Duration::from_millis(90_000)
        );
        let expired = started.checked_sub(Duration::from_secs(120)).unwrap();
        assert!(command.anchor_original_window(expired).is_err());
    }
}
