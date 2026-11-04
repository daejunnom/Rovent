//! Pure utility-only CLI parsing. Parsing does not open pinned files, admit a
//! cohort, create a cgroup, import a model or issue execution authority.
//!
//! The concrete utility runtime must join these explicit caller bounds to the
//! original utility registration/window. CPU, memory, affinity, device and the
//! aggregate-quota policy come from that exact registration, never from a
//! recovered P/C/V request or an environment/default fallback.
#![cfg(target_os = "linux")]

use super::execution::{
    ROVENT_UTILITY_MAX_CLEANUP_NS, ROVENT_UTILITY_MAX_READ_BYTES, ROVENT_UTILITY_MAX_WALL_NS,
};
use crate::ArenaError;
use crate::rovent_training_process::{MAX_TRAINER_OUTPUT_BYTES, RoventTrainingFilePin};
use rz_experiments::pals_data_v3::generation_sha;
use std::{
    collections::BTreeMap,
    ffi::OsString,
    path::{Component, Path, PathBuf},
    time::{Duration, Instant},
};

pub const ROVENT_UTILITY_CLI_USAGE: &str = "rz-rovent-utility-trainer --action <inspect|execute> --registration <absolute.json> --registration-sha256 <sha256> --registration-bytes <bytes> --wall-ms <ms> --cleanup-ms <ms> --read-bytes <bytes> --output-bytes <bytes> --stdio-bytes <bytes> --delegated-cgroup <absolute-directory> --cgroup-leaf <new-leaf>";

const MAX_REGISTRATION_BYTES: u64 = 16 << 20;
const MAX_STDIO_BYTES: u64 = 64 << 10;
const MAX_PATH_BYTES: usize = 4096;
const OPTIONS: [&str; 11] = [
    "--action",
    "--registration",
    "--registration-sha256",
    "--registration-bytes",
    "--wall-ms",
    "--cleanup-ms",
    "--read-bytes",
    "--output-bytes",
    "--stdio-bytes",
    "--delegated-cgroup",
    "--cgroup-leaf",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoventUtilityCliAction {
    /// Read-only exact utility metadata/cohort inspection; no execution lease.
    Inspect,
    /// Request the registration's own utility Train or Export action.
    Execute,
}

/// Caller requests only. An absolute delegated path is not an owned cgroup FD,
/// and a leaf spelling does not prove that a new isolated scope was created.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RoventUtilityCliScopeRequest {
    pub wall_ms: u64,
    pub cleanup_ms: u64,
    pub max_read_bytes: u64,
    pub max_output_bytes: u64,
    pub max_stdio_bytes: u64,
    pub delegated_cgroup: PathBuf,
    pub cgroup_leaf: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RoventUtilityCliCommand {
    pub action: RoventUtilityCliAction,
    pub registration: RoventTrainingFilePin,
    pub scope: RoventUtilityCliScopeRequest,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RoventUtilityCliParse {
    Help,
    Command(RoventUtilityCliCommand),
}

/// Unadmitted request anchored once to the thin binary's original entry time.
/// Neither a retry nor a metadata read can issue a replacement execution window.
pub struct RoventUtilityCliAdmissionRequest {
    command: RoventUtilityCliCommand,
    original_started: Instant,
    execution_deadline: Instant,
    whole_deadline: Instant,
}

impl RoventUtilityCliCommand {
    pub fn validate(&self) -> Result<(), ArenaError> {
        let scope = &self.scope;
        require(
            exact_absolute_path(&self.registration.path)
                && self
                    .registration
                    .path
                    .extension()
                    .is_some_and(|value| value == "json")
                && generation_sha(&self.registration.sha256)
                && (1..=MAX_REGISTRATION_BYTES).contains(&self.registration.bytes),
            "exact absolute registration JSON path, SHA and byte count required",
        )?;
        require(
            (1..=ROVENT_UTILITY_MAX_WALL_NS / 1_000_000).contains(&scope.wall_ms)
                && (1..=ROVENT_UTILITY_MAX_CLEANUP_NS / 1_000_000).contains(&scope.cleanup_ms)
                && scope.wall_ms.checked_add(scope.cleanup_ms).is_some()
                && scope.max_read_bytes >= self.registration.bytes
                && scope.max_read_bytes <= ROVENT_UTILITY_MAX_READ_BYTES
                && (1..=MAX_TRAINER_OUTPUT_BYTES).contains(&scope.max_output_bytes)
                && (1..=MAX_STDIO_BYTES).contains(&scope.max_stdio_bytes),
            "finite canonical utility time/read/artifact/stdio bounds required",
        )?;
        require(
            exact_absolute_path(&scope.delegated_cgroup) && valid_leaf(&scope.cgroup_leaf),
            "absolute delegated cgroup and a single new leaf name required",
        )
    }

    pub fn anchor_original_window(
        self,
        original_started: Instant,
    ) -> Result<RoventUtilityCliAdmissionRequest, ArenaError> {
        self.validate()?;
        let execution_deadline = original_started
            .checked_add(Duration::from_millis(self.scope.wall_ms))
            .ok_or_else(|| invalid("execution deadline is not representable"))?;
        let whole_deadline = execution_deadline
            .checked_add(Duration::from_millis(self.scope.cleanup_ms))
            .ok_or_else(|| invalid("cleanup deadline is not representable"))?;
        let now = Instant::now();
        require(
            original_started <= now && now < execution_deadline,
            "original execution window expired or starts in the future",
        )?;
        Ok(RoventUtilityCliAdmissionRequest {
            command: self,
            original_started,
            execution_deadline,
            whole_deadline,
        })
    }
}

impl RoventUtilityCliAdmissionRequest {
    pub fn command(&self) -> &RoventUtilityCliCommand {
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
            "original admission window expired; no fresh window was issued",
        )
    }
}

/// Parse args_os().skip(1). Exactly eleven options are required, each once,
/// with separate UTF-8 values. Error messages never echo raw caller arguments.
pub fn parse_rovent_utility_cli(
    arguments: impl IntoIterator<Item = OsString>,
) -> Result<RoventUtilityCliParse, ArenaError> {
    let arguments: Vec<OsString> = arguments.into_iter().take(OPTIONS.len() * 2 + 1).collect();
    if arguments.len() == 1 && arguments[0] == "--help" {
        return Ok(RoventUtilityCliParse::Help);
    }
    require(
        arguments.len() == OPTIONS.len() * 2,
        "each of the eleven documented options is required exactly once; use --help",
    )?;
    let mut values = BTreeMap::new();
    for pair in arguments.chunks_exact(2) {
        let option = pair[0]
            .to_str()
            .ok_or_else(|| invalid("option is not UTF-8"))?;
        let value = pair[1]
            .to_str()
            .filter(|value| !value.is_empty() && value.len() <= MAX_PATH_BYTES)
            .ok_or_else(|| invalid("value is empty, not UTF-8 or exceeds its bound"))?;
        require(OPTIONS.contains(&option), "unknown option")?;
        require(
            values.insert(option, value).is_none(),
            "option was supplied more than once",
        )?;
    }
    let value = |key| {
        values
            .get(key)
            .copied()
            .ok_or_else(|| invalid("required option is missing"))
    };
    let action = match value("--action")? {
        "inspect" => RoventUtilityCliAction::Inspect,
        "execute" => RoventUtilityCliAction::Execute,
        _ => return Err(invalid("action must be inspect or execute")),
    };
    let number = |key| canonical_u64(value(key)?);
    let command = RoventUtilityCliCommand {
        action,
        registration: RoventTrainingFilePin {
            path: PathBuf::from(value("--registration")?),
            sha256: value("--registration-sha256")?.into(),
            bytes: number("--registration-bytes")?,
        },
        scope: RoventUtilityCliScopeRequest {
            wall_ms: number("--wall-ms")?,
            cleanup_ms: number("--cleanup-ms")?,
            max_read_bytes: number("--read-bytes")?,
            max_output_bytes: number("--output-bytes")?,
            max_stdio_bytes: number("--stdio-bytes")?,
            delegated_cgroup: PathBuf::from(value("--delegated-cgroup")?),
            cgroup_leaf: value("--cgroup-leaf")?.into(),
        },
    };
    command.validate()?;
    Ok(RoventUtilityCliParse::Command(command))
}

fn invalid(reason: &str) -> ArenaError {
    ArenaError::Integrity(format!("Rovent utility CLI: {reason}"))
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
        "integer must use canonical unsigned decimal notation",
    )?;
    value.parse().map_err(|_| invalid("integer exceeds u64"))
}
fn exact_absolute_path(path: &Path) -> bool {
    path.is_absolute()
        && path.as_os_str().len() <= MAX_PATH_BYTES
        && path.components().count() <= 64
        && path.components().all(|part| match part {
            Component::RootDir => true,
            Component::Normal(name) => name
                .to_str()
                .is_some_and(|name| !name.chars().any(char::is_control)),
            _ => false,
        })
        && path.components().collect::<PathBuf>().as_os_str() == path.as_os_str()
}
fn valid_leaf(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value.as_bytes()[0].is_ascii_alphanumeric()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arguments() -> Vec<OsString> {
        [
            ("--action", "inspect".to_owned()),
            ("--registration", "/utility/registration.json".to_owned()),
            ("--registration-sha256", "1".repeat(64)),
            ("--registration-bytes", "512".to_owned()),
            ("--wall-ms", "60000".to_owned()),
            ("--cleanup-ms", "30000".to_owned()),
            ("--read-bytes", "4096".to_owned()),
            ("--output-bytes", "1048576".to_owned()),
            ("--stdio-bytes", "65536".to_owned()),
            ("--delegated-cgroup", "/utility/delegated".to_owned()),
            ("--cgroup-leaf", "utility-window-1".to_owned()),
        ]
        .into_iter()
        .flat_map(|(key, value)| [OsString::from(key), OsString::from(value)])
        .collect()
    }
    fn replace(arguments: &mut [OsString], key: &str, value: &str) {
        let slot = arguments
            .iter()
            .position(|argument| argument == key)
            .unwrap();
        arguments[slot + 1] = value.into();
    }

    #[test]
    fn standalone_help_is_the_only_request_without_eleven_options() {
        assert_eq!(OPTIONS.len(), 11);
        assert_eq!(arguments().len(), 22);
        assert!(matches!(
            parse_rovent_utility_cli([OsString::from("--help")]).unwrap(),
            RoventUtilityCliParse::Help
        ));
        assert!(parse_rovent_utility_cli(Vec::<OsString>::new()).is_err());
        assert!(parse_rovent_utility_cli(["--help".into(), "inspect".into()]).is_err());
    }

    #[test]
    fn inspect_and_execute_do_not_replace_the_registrations_train_export_action() {
        for action in ["inspect", "execute"] {
            let mut input = arguments();
            replace(&mut input, "--action", action);
            let RoventUtilityCliParse::Command(command) = parse_rovent_utility_cli(input).unwrap()
            else {
                panic!("explicit utility command required")
            };
            assert_eq!(command.registration.bytes, 512);
            assert_eq!(command.scope.cgroup_leaf, "utility-window-1");
            assert_eq!(
                command.action,
                if action == "inspect" {
                    RoventUtilityCliAction::Inspect
                } else {
                    RoventUtilityCliAction::Execute
                }
            );
        }
        let mut retag = arguments();
        replace(&mut retag, "--action", "train");
        assert!(parse_rovent_utility_cli(retag).is_err());
    }

    #[test]
    fn duplicate_missing_unknown_and_extra_options_are_rejected() {
        let mut duplicate = arguments();
        duplicate[2] = "--action".into();
        assert!(parse_rovent_utility_cli(duplicate).is_err());
        let mut missing = arguments();
        missing.truncate(missing.len() - 2);
        assert!(parse_rovent_utility_cli(missing).is_err());
        let mut unknown = arguments();
        unknown[0] = "--device".into();
        assert!(parse_rovent_utility_cli(unknown).is_err());
        let mut extra = arguments();
        extra.extend(["--device".into(), "cpu".into()]);
        assert!(parse_rovent_utility_cli(extra).is_err());
    }

    #[test]
    fn malformed_or_inconsistent_bounds_do_not_reach_admission() {
        for (key, value) in [
            ("--wall-ms", "0"),
            ("--wall-ms", "060000"),
            ("--wall-ms", "86400001"),
            ("--wall-ms", "18446744073709551616"),
            ("--cleanup-ms", "0"),
            ("--cleanup-ms", "30001"),
            ("--read-bytes", "511"),
            ("--read-bytes", "+4096"),
            ("--output-bytes", "0"),
            ("--output-bytes", "19327352833"),
            ("--stdio-bytes", "65537"),
            ("--registration-sha256", "unknown"),
            ("--registration-bytes", "16777217"),
        ] {
            let mut input = arguments();
            replace(&mut input, key, value);
            assert!(parse_rovent_utility_cli(input).is_err(), "{key}");
        }
    }

    #[test]
    fn paths_are_exact_and_leaf_cannot_escape_its_original_scope() {
        for (key, value) in [
            ("--registration", "registration.json"),
            ("--registration", "/utility/../registration.json"),
            ("--registration", "/utility/./registration.json"),
            ("--registration", "/utility//registration.json"),
            ("--registration", "/utility/registration.json/"),
            ("--registration", "/utility/registration.bin"),
            ("--delegated-cgroup", "delegated"),
            ("--delegated-cgroup", "/utility/delegated/"),
            ("--cgroup-leaf", "."),
            ("--cgroup-leaf", ".."),
            ("--cgroup-leaf", "first/second"),
            ("--cgroup-leaf", "first\\second"),
            ("--cgroup-leaf", "leaf:stream"),
        ] {
            let mut input = arguments();
            replace(&mut input, key, value);
            assert!(parse_rovent_utility_cli(input).is_err(), "{key}");
        }
    }

    #[test]
    fn original_deadlines_are_fixed_and_an_expired_window_is_not_refreshed() {
        let RoventUtilityCliParse::Command(command) =
            parse_rovent_utility_cli(arguments()).unwrap()
        else {
            panic!("explicit utility command required")
        };
        let started = Instant::now();
        let request = command.clone().anchor_original_window(started).unwrap();
        assert_eq!(request.original_started(), started);
        assert_eq!(request.command(), &command);
        assert_eq!(
            request.execution_deadline().duration_since(started),
            Duration::from_millis(60_000)
        );
        assert_eq!(
            request.whole_deadline().duration_since(started),
            Duration::from_millis(90_000)
        );
        let expired = started.checked_sub(Duration::from_secs(120)).unwrap();
        assert!(command.clone().anchor_original_window(expired).is_err());
        let future = started.checked_add(Duration::from_secs(120)).unwrap();
        assert!(command.anchor_original_window(future).is_err());
    }
}
