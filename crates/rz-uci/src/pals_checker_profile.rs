//! Pinned, declaration-only CPU_R profile for the first Stockfish selection.
//!
//! This is deliberately `stockfish_embedded_nnue` only. It is not a general
//! external-engine launcher. Loading, registration and checker creation never
//! spawn a process. The caller owns the finite `start` clock and actual evidence.
//! A declared embedded model, public source or license does not prove loading,
//! training, rights, precision, or that a requested UCI option was applied.

#[cfg(unix)]
use cap_fs_ext::OpenOptionsSyncExt;
use cap_fs_ext::{DirExt, FollowSymlinks, OpenOptionsFollowExt};
use cap_std::{
    ambient_authority,
    fs::{Dir, OpenOptions},
};
use rz_search::cpu_checker::{CheckerError, ExternalCheckerIdentity};
use rz_search::external_cpu::{
    EXTERNAL_UCI_SCORE_SEMANTICS, ExternalCpuConfig, ExternalUciCpuChecker, arguments_sha256,
};
use serde::de::{Error as _, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::fs::Metadata;
use std::io::{Read, Write};
#[cfg(unix)]
use std::os::unix::fs::MetadataExt;
#[cfg(windows)]
use std::os::windows::fs::MetadataExt;
use std::path::{Component, Path, PathBuf};
#[cfg(target_os = "linux")]
use std::time::Duration;

pub mod rovent;

pub const PALS_CHECKER_PROFILE_SCHEMA: &str = "rz-pals-external-checker-profile/1";
pub const PALS_CHECKER_REGISTRATION_SCHEMA: &str = "rz-pals-external-checker-registration/1";
pub const PALS_ADAPTIVE_CHECKER_PROFILE_V5_SCHEMA: &str =
    "rz-pals-adaptive-external-checker-profile-v5/1";
pub const MAX_CHECKER_PROFILE_BYTES: usize = 64 * 1024;
const CANONICAL_DOMAIN: &str = "rz-pals-external-checker-profile-canonical/1";
const ADAPTIVE_CANONICAL_DOMAIN: &str = "rz-pals-adaptive-external-checker-profile-v5-canonical/1";
const STOCKFISH_SOURCE: &str = "https://github.com/official-stockfish/Stockfish";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PalsCheckerSelection {
    StockfishEmbeddedNnue,
}

/// Fields are declarations. Actual executable/ELF/argv verification and the
/// supported option/name handshake remain owned by `ExternalUciCpuChecker`.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PalsCheckerProfile {
    pub schema: String,
    pub selection: PalsCheckerSelection,
    /// Absolute Linux path; no shell, interpreter, or PATH lookup is admitted.
    pub program: String,
    pub working_directory: String,
    /// The first selection accepts no engine command-line arguments.
    pub arguments: Vec<String>,
    pub identity: ExternalCheckerIdentity,
    pub expected_uci_name: String,
    pub max_depth: u16,
    pub max_prefix_plies: usize,
    pub handshake_timeout_ms: u64,
    pub max_task_wall_time_ms: u64,
    pub stop_grace_ms: u64,
    pub shutdown_grace_ms: u64,
    pub max_output_bytes: usize,
    pub max_line_bytes: usize,
}

/// Only `load` can construct this value: raw file and semantic identities are
/// kept separate, and the declaration cannot be modified after verification.
#[derive(Debug)]
pub struct LoadedPalsCheckerProfile {
    profile: PalsCheckerProfile,
    file_sha256: String,
    canonical_sha256: String,
    file_bytes: u64,
}

/// V5 Full options have a different schema and canonical domain. This cannot
/// be passed to the legacy checker/config consumer or constructed without the
/// exact byte pin; it still attests only declaration admission, never spawn.
#[derive(Debug)]
pub struct LoadedPalsAdaptiveCheckerProfileV5 {
    profile: PalsCheckerProfile,
    file_sha256: String,
    canonical_sha256: String,
    file_bytes: u64,
}

#[derive(Debug, Serialize)]
pub struct PalsCheckerProfileRegistration<'a> {
    pub schema: &'static str,
    pub file_sha256: &'a str,
    pub canonical_sha256: &'a str,
    pub file_bytes: u64,
    pub declaration: &'a PalsCheckerProfile,
    pub declaration_only: bool,
    /// This module did not start a process. A caller's later start is separate.
    pub process_start_performed: bool,
    pub observed_uci_name: Option<&'a str>,
    pub options_application_observed: Option<bool>,
    pub model_loading_observed: Option<bool>,
}

fn invalid(detail: &'static str) -> CheckerError {
    CheckerError::Invalid(detail)
}
fn file_error(code: &'static str) -> CheckerError {
    CheckerError::External {
        stage: "profile",
        code,
    }
}
fn sha(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn stockfish_name(value: &str) -> bool {
    value == "Stockfish" || value.starts_with("Stockfish ")
}
fn linux_path(value: &str, root_allowed: bool) -> bool {
    if value == "/" {
        return root_allowed;
    }
    value.starts_with('/')
        && value.len() <= 4096
        && !value.contains('\\')
        && !value.chars().any(char::is_control)
        && value
            .split('/')
            .skip(1)
            .all(|part| !part.is_empty() && part != "." && part != "..")
}

impl PalsCheckerProfile {
    /// Parse exact caller-pinned bytes without reopening a path. The caller
    /// separately proves retained descriptor identity; this pure admission
    /// validates byte size/hash, duplicate keys, schema and immutable meaning.
    pub fn from_verified_adaptive_bytes(
        bytes: &[u8],
        expected_file_sha256: &str,
    ) -> Result<LoadedPalsAdaptiveCheckerProfileV5, CheckerError> {
        let (profile, file_sha256) = parse_verified_profile_bytes(bytes, expected_file_sha256)?;
        profile.validate_adaptive_v5()?;
        let canonical_sha256 = profile.canonical_digest_for(ADAPTIVE_CANONICAL_DOMAIN)?;
        Ok(LoadedPalsAdaptiveCheckerProfileV5 {
            profile,
            file_sha256,
            canonical_sha256,
            file_bytes: bytes.len() as u64,
        })
    }

    /// Read at most 64 KiB from a pinned regular file. Every path component is
    /// opened without following links. Parsing uses these exact hashed bytes.
    /// This never reads the engine, a model, or starts a subprocess.
    pub fn load(
        path: &Path,
        expected_file_sha256: &str,
    ) -> Result<LoadedPalsCheckerProfile, CheckerError> {
        if !sha(expected_file_sha256) {
            return Err(invalid("checker profile requires lowercase file SHA256"));
        }
        let (directory, name) = profile_parent(path)?;
        let mut options = OpenOptions::new();
        options.read(true).follow(FollowSymlinks::No);
        // A nonregular FIFO must not block before its metadata can be rejected.
        // Regular-file reads can still wait for storage; this is a byte cap,
        // not an independently measured startup wall-time guarantee.
        #[cfg(unix)]
        options.nonblock(true);
        let mut file = directory
            .open_with(&name, &options)
            .map_err(|_| file_error("file_open_nofollow"))?
            .into_std();
        let before = file.metadata().map_err(|_| file_error("file_metadata"))?;
        if !before.is_file() || before.len() == 0 || before.len() > MAX_CHECKER_PROFILE_BYTES as u64
        {
            return Err(invalid(
                "checker profile must be a nonempty bounded regular file",
            ));
        }
        let mut bytes = Vec::with_capacity(before.len() as usize);
        (&mut file)
            .take(MAX_CHECKER_PROFILE_BYTES as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(|_| file_error("file_read"))?;
        let after = file.metadata().map_err(|_| file_error("file_recheck"))?;
        let current = directory
            .open_with(&name, &options)
            .map_err(|_| file_error("path_recheck_nofollow"))?
            .into_std()
            .metadata()
            .map_err(|_| file_error("path_metadata"))?;
        if bytes.len() as u64 != before.len()
            || bytes.len() > MAX_CHECKER_PROFILE_BYTES
            || !same_metadata(&before, &after)
            || !same_metadata(&after, &current)
        {
            return Err(invalid("checker profile changed during bounded read"));
        }
        let (profile, file_sha256) = parse_verified_profile_bytes(&bytes, expected_file_sha256)?;
        profile.validate()?;
        let canonical_sha256 = profile.canonical_digest()?;
        Ok(LoadedPalsCheckerProfile {
            profile,
            file_sha256,
            canonical_sha256,
            file_bytes: bytes.len() as u64,
        })
    }

    pub fn validate(&self) -> Result<(), CheckerError> {
        self.validate_common(PALS_CHECKER_PROFILE_SCHEMA)?;
        validate_options(&self.identity.options)
    }

    fn validate_common(&self, expected_schema: &str) -> Result<(), CheckerError> {
        if self.schema != expected_schema {
            return Err(invalid("unsupported checker profile schema"));
        }
        if !self.identity.assets.is_empty() {
            return Err(CheckerError::Unsupported(
                "stockfish_embedded_nnue rejects separately loaded assets",
            ));
        }
        if !self.arguments.is_empty() {
            return Err(CheckerError::Unsupported(
                "stockfish_embedded_nnue admits no engine argv",
            ));
        }
        self.identity.validate()?;
        if self.identity.adapter_semantics != EXTERNAL_UCI_SCORE_SEMANTICS
            || self.identity.launch_arguments_sha256 != arguments_sha256(&[])
            || !stockfish_name(&self.identity.declared_name)
            || !(self.identity.declared_source == STOCKFISH_SOURCE
                || self
                    .identity
                    .declared_source
                    .starts_with(&format!("{STOCKFISH_SOURCE}/")))
            || self.identity.declared_license != "GPL-3.0-or-later"
        {
            return Err(invalid(
                "Stockfish checker declaration or argument identity mismatch",
            ));
        }
        let model = &self.identity.model_metadata;
        if model.weights_sha256.is_some()
            || model.precision.is_some()
            || model.declared_rights.is_some()
        {
            return Err(invalid("embedded NNUE model metadata must remain unknown"));
        }
        if !linux_path(&self.program, false) || !linux_path(&self.working_directory, true) {
            return Err(invalid(
                "checker binary and cwd require bounded absolute Linux paths",
            ));
        }
        if self.expected_uci_name.is_empty()
            || self.expected_uci_name.len() > 256
            || self.expected_uci_name.chars().any(char::is_control)
            || !stockfish_name(&self.expected_uci_name)
        {
            return Err(invalid(
                "checker profile requires exact expected Stockfish UCI name",
            ));
        }
        if self.max_depth == 0 || self.max_depth > 64 || self.max_prefix_plies > 4096 {
            return Err(invalid("checker profile depth/history cap"));
        }
        if [
            self.handshake_timeout_ms,
            self.max_task_wall_time_ms,
            self.stop_grace_ms,
            self.shutdown_grace_ms,
        ]
        .into_iter()
        .any(|time| time == 0 || time > 180_000)
        {
            return Err(invalid(
                "checker profile requires finite handshake/task/stop/shutdown budgets",
            ));
        }
        if self.max_output_bytes == 0
            || self.max_output_bytes > 16 * 1024 * 1024
            || self.max_line_bytes == 0
            || self.max_line_bytes > 65_536
            || self.max_line_bytes > self.max_output_bytes
        {
            return Err(invalid("checker profile lifetime output/line cap"));
        }
        Ok(())
    }

    fn validate_adaptive_v5(&self) -> Result<(), CheckerError> {
        self.validate_common(PALS_ADAPTIVE_CHECKER_PROFILE_V5_SCHEMA)?;
        use rz_search::external_cpu::{EXTERNAL_STOP_DRAIN_RESERVE, STOCKFISH19_SOURCE_COMMIT};
        let expected = std::collections::BTreeMap::from([
            ("Skill Level".into(), "20".into()),
            ("UCI_LimitStrength".into(), "false".into()),
            ("Threads".into(), "1".into()),
            ("Hash".into(), "256".into()),
            ("Ponder".into(), "false".into()),
            ("NumaPolicy".into(), "none".into()),
            ("SyzygyPath".into(), String::new()),
            ("MultiPV".into(), "1".into()),
        ]);
        let pinned_source = [
            format!("{STOCKFISH_SOURCE}/tree/{STOCKFISH19_SOURCE_COMMIT}"),
            format!("{STOCKFISH_SOURCE}/commit/{STOCKFISH19_SOURCE_COMMIT}"),
        ];
        if self.identity.options != expected
            || self.identity.declared_version != "19"
            || self.identity.declared_name != "Stockfish 19"
            || self.expected_uci_name != "Stockfish 19"
            || !pinned_source.contains(&self.identity.declared_source)
            || self.stop_grace_ms != EXTERNAL_STOP_DRAIN_RESERVE.as_millis() as u64
            || self.max_task_wall_time_ms < 600
        {
            return Err(invalid(
                "adaptive Stockfish 19 Full pin/options/stop reserve differ",
            ));
        }
        Ok(())
    }

    fn canonical_digest(&self) -> Result<String, CheckerError> {
        self.validate()?;
        self.canonical_digest_for(CANONICAL_DOMAIN)
    }

    fn canonical_digest_for(&self, domain: &str) -> Result<String, CheckerError> {
        let value = sorted_value(
            serde_json::to_value(self)
                .map_err(|_| invalid("cannot encode checker profile declaration"))?,
        );
        let mut output = BoundedJson { bytes: Vec::new() };
        serde_json::to_writer(&mut output, &(domain, value))
            .map_err(|_| invalid("canonical checker profile exceeds byte bound"))?;
        Ok(format!("{:x}", Sha256::digest(&output.bytes)))
    }

    #[cfg(target_os = "linux")]
    fn external_config(&self) -> ExternalCpuConfig {
        ExternalCpuConfig {
            program: PathBuf::from(&self.program),
            arguments: self
                .arguments
                .iter()
                .map(|arg| OsString::from(arg.as_str()))
                .collect(),
            working_directory: PathBuf::from(&self.working_directory),
            identity: self.identity.clone(),
            expected_uci_name: Some(self.expected_uci_name.clone()),
            max_depth: self.max_depth,
            max_prefix_plies: self.max_prefix_plies,
            handshake_timeout: Duration::from_millis(self.handshake_timeout_ms),
            max_task_wall_time: Duration::from_millis(self.max_task_wall_time_ms),
            stop_grace: Duration::from_millis(self.stop_grace_ms),
            shutdown_grace: Duration::from_millis(self.shutdown_grace_ms),
            max_output_bytes: self.max_output_bytes,
            max_line_bytes: self.max_line_bytes,
        }
    }
}

fn parse_verified_profile_bytes(
    bytes: &[u8],
    expected_file_sha256: &str,
) -> Result<(PalsCheckerProfile, String), CheckerError> {
    if !sha(expected_file_sha256) {
        return Err(invalid("checker profile requires lowercase file SHA256"));
    }
    if bytes.is_empty() || bytes.len() > MAX_CHECKER_PROFILE_BYTES {
        return Err(invalid(
            "checker profile must be a nonempty bounded regular file",
        ));
    }
    let file_sha256 = format!("{:x}", Sha256::digest(bytes));
    if file_sha256 != expected_file_sha256 {
        return Err(invalid("checker profile file SHA256 mismatch"));
    }
    // BTreeMap would silently keep the last repeated option. Keep this check
    // shared across schemas, before the unchanged typed JSON representation.
    serde_json::from_slice::<UniqueJsonKeys>(bytes)
        .map_err(|_| invalid("invalid JSON or duplicate checker profile object key"))?;
    let profile = serde_json::from_slice(bytes)
        .map_err(|_| invalid("invalid stockfish_embedded_nnue profile JSON"))?;
    Ok((profile, file_sha256))
}

impl LoadedPalsAdaptiveCheckerProfileV5 {
    pub fn profile(&self) -> &PalsCheckerProfile {
        &self.profile
    }
    pub fn file_sha256(&self) -> &str {
        &self.file_sha256
    }
    pub fn canonical_sha256(&self) -> &str {
        &self.canonical_sha256
    }
    pub fn file_bytes(&self) -> u64 {
        self.file_bytes
    }
    pub fn adaptive_config(&self) -> Result<ExternalCpuConfig, CheckerError> {
        self.profile.validate_adaptive_v5()?;
        #[cfg(target_os = "linux")]
        {
            let config = self.profile.external_config();
            config.validate()?;
            Ok(config)
        }
        #[cfg(not(target_os = "linux"))]
        Err(CheckerError::Unsupported(
            "adaptive external CPU execution requires Linux",
        ))
    }
}

impl LoadedPalsCheckerProfile {
    pub fn profile(&self) -> &PalsCheckerProfile {
        &self.profile
    }
    pub fn file_sha256(&self) -> &str {
        &self.file_sha256
    }
    pub fn canonical_sha256(&self) -> &str {
        &self.canonical_sha256
    }
    pub fn registration(&self) -> PalsCheckerProfileRegistration<'_> {
        PalsCheckerProfileRegistration {
            schema: PALS_CHECKER_REGISTRATION_SCHEMA,
            file_sha256: &self.file_sha256,
            canonical_sha256: &self.canonical_sha256,
            file_bytes: self.file_bytes,
            declaration: &self.profile,
            declaration_only: true,
            process_start_performed: false,
            observed_uci_name: None,
            options_application_observed: None,
            model_loading_observed: None,
        }
    }
    /// Validate configuration only. Executable pinning and startup remain on
    /// the caller's finite clock; unsupported platforms never use a fallback.
    pub fn config(&self) -> Result<ExternalCpuConfig, CheckerError> {
        self.profile.validate()?;
        #[cfg(target_os = "linux")]
        {
            let config = self.profile.external_config();
            config.validate()?;
            Ok(config)
        }
        #[cfg(not(target_os = "linux"))]
        Err(CheckerError::Unsupported(
            "external CPU_R execution requires Linux",
        ))
    }
    /// Creates the existing unstarted owner. Call `start` explicitly and keep
    /// its failed-attempt diagnostics and bounded shutdown evidence.
    pub fn create_checker(&self) -> Result<ExternalUciCpuChecker, CheckerError> {
        ExternalUciCpuChecker::create(self.config()?)
    }
}

fn validate_options(options: &BTreeMap<String, String>) -> Result<(), CheckerError> {
    const ALLOWED: [&str; 8] = [
        "Threads",
        "Hash",
        "Ponder",
        "UCI_Chess960",
        "MultiPV",
        "SyzygyPath",
        "SyzygyProbeLimit",
        "UCI_ShowWDL",
    ];
    if options.keys().any(|name| !ALLOWED.contains(&name.as_str())) {
        return Err(CheckerError::Unsupported(
            "stockfish_embedded_nnue option is outside the fixed whitelist",
        ));
    }
    for (name, expected) in [
        ("Ponder", "false"),
        ("UCI_Chess960", "false"),
        ("MultiPV", "1"),
        ("SyzygyPath", ""),
        ("SyzygyProbeLimit", "0"),
    ] {
        if options.get(name).map(String::as_str) != Some(expected) {
            return Err(invalid(
                "Stockfish profile must disable pondering/Chess960/tablebases and use MultiPV=1",
            ));
        }
    }
    if !matches!(options.get("Threads").map(String::as_str), Some("1" | "2")) {
        return Err(invalid("Stockfish profile Threads must be 1 or 2"));
    }
    let hash = options
        .get("Hash")
        .ok_or_else(|| invalid("Stockfish profile requires bounded Hash MiB"))?;
    let hash_mib: u16 = hash
        .parse()
        .map_err(|_| invalid("invalid Stockfish Hash MiB"))?;
    if !(1..=1024).contains(&hash_mib) || *hash != hash_mib.to_string() {
        return Err(invalid(
            "Stockfish Hash MiB must be canonical decimal in 1..=1024",
        ));
    }
    if options
        .get("UCI_ShowWDL")
        .is_some_and(|value| value != "true" && value != "false")
    {
        return Err(invalid("Stockfish UCI_ShowWDL must be an explicit boolean"));
    }
    Ok(())
}

/// Pin the root and each parent component instead of resolving symlink parents
/// through an ambient open. Traversal and nonabsolute profile paths are rejected.
fn profile_parent(path: &Path) -> Result<(Dir, OsString), CheckerError> {
    if !path.is_absolute() || path.as_os_str().as_encoded_bytes().len() > 4096 {
        return Err(invalid("checker profile path must be bounded and absolute"));
    }
    let mut root = PathBuf::new();
    let mut names = Vec::new();
    for component in path.components() {
        match component {
            Component::Prefix(_) | Component::RootDir if names.is_empty() => {
                root.push(component.as_os_str())
            }
            Component::Normal(name) if names.len() < 128 => names.push(name.to_os_string()),
            _ => {
                return Err(invalid(
                    "checker profile path has traversal or too many components",
                ));
            }
        }
    }
    let name = names
        .pop()
        .ok_or_else(|| invalid("checker profile path requires a filename"))?;
    let mut directory =
        Dir::open_ambient_dir(&root, ambient_authority()).map_err(|_| file_error("root_open"))?;
    for parent in names {
        directory = directory
            .open_dir_nofollow(&parent)
            .map_err(|_| file_error("parent_open_nofollow"))?;
    }
    Ok((directory, name))
}

fn same_metadata(before: &Metadata, after: &Metadata) -> bool {
    if !after.is_file()
        || before.len() != after.len()
        || before.modified().ok() != after.modified().ok()
    {
        return false;
    }
    #[cfg(unix)]
    if before.dev() != after.dev()
        || before.ino() != after.ino()
        || before.mode() != after.mode()
        || before.nlink() != after.nlink()
        || before.ctime() != after.ctime()
        || before.ctime_nsec() != after.ctime_nsec()
    {
        return false;
    }
    #[cfg(windows)]
    if before.creation_time() != after.creation_time()
        || before.file_attributes() != after.file_attributes()
    {
        return false;
    }
    true
}

/// A validation-only pass. Scalar values are discarded and each object's key
/// set is bounded by the already enforced 64 KiB file cap. serde_json retains
/// its normal nesting limit; no recursive model/profile interpretation occurs.
struct UniqueJsonKeys;
impl<'de> Deserialize<'de> for UniqueJsonKeys {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(UniqueJsonKeyVisitor)
    }
}
struct UniqueJsonKeyVisitor;
impl<'de> Visitor<'de> for UniqueJsonKeyVisitor {
    type Value = UniqueJsonKeys;
    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("JSON with unique decoded object keys")
    }
    fn visit_unit<E: serde::de::Error>(self) -> Result<Self::Value, E> {
        Ok(UniqueJsonKeys)
    }
    fn visit_bool<E: serde::de::Error>(self, _value: bool) -> Result<Self::Value, E> {
        Ok(UniqueJsonKeys)
    }
    fn visit_i64<E: serde::de::Error>(self, _value: i64) -> Result<Self::Value, E> {
        Ok(UniqueJsonKeys)
    }
    fn visit_u64<E: serde::de::Error>(self, _value: u64) -> Result<Self::Value, E> {
        Ok(UniqueJsonKeys)
    }
    fn visit_f64<E: serde::de::Error>(self, _value: f64) -> Result<Self::Value, E> {
        Ok(UniqueJsonKeys)
    }
    fn visit_str<E: serde::de::Error>(self, _value: &str) -> Result<Self::Value, E> {
        Ok(UniqueJsonKeys)
    }
    fn visit_string<E: serde::de::Error>(self, _value: String) -> Result<Self::Value, E> {
        Ok(UniqueJsonKeys)
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<Self::Value, A::Error> {
        while sequence.next_element::<UniqueJsonKeys>()?.is_some() {}
        Ok(UniqueJsonKeys)
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut seen = BTreeSet::new();
        while let Some(key) = map.next_key::<String>()? {
            if !seen.insert(key) {
                return Err(A::Error::custom("duplicate decoded JSON object key"));
            }
            map.next_value::<UniqueJsonKeys>()?;
        }
        Ok(UniqueJsonKeys)
    }
}

fn sorted_value(value: Value) -> Value {
    match value {
        Value::Object(fields) => Value::Object(
            fields
                .into_iter()
                .map(|(key, value)| (key, sorted_value(value)))
                .collect::<BTreeMap<_, _>>()
                .into_iter()
                .collect(),
        ),
        Value::Array(items) => Value::Array(items.into_iter().map(sorted_value).collect()),
        other => other,
    }
}
struct BoundedJson {
    bytes: Vec<u8>,
}
impl Write for BoundedJson {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > MAX_CHECKER_PROFILE_BYTES.saturating_sub(self.bytes.len()) {
            return Err(std::io::Error::other(
                "checker canonical profile byte limit",
            ));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rz_search::cpu_checker::{ExternalModelMetadata, ExternalTrainingKnowledge};
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT_TEMP: AtomicU64 = AtomicU64::new(1);
    struct OwnedTemp(PathBuf);
    impl OwnedTemp {
        fn new() -> Self {
            let root = std::env::temp_dir().join(format!(
                "rz-pals-checker-profile-test-{}-{}",
                std::process::id(),
                NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&root).unwrap();
            Self(root)
        }
        fn write(&self, name: &str, bytes: &[u8]) -> (PathBuf, String) {
            let path = self.0.join(name);
            std::fs::write(&path, bytes).unwrap();
            (path, format!("{:x}", Sha256::digest(bytes)))
        }
    }
    impl Drop for OwnedTemp {
        fn drop(&mut self) {
            let (Ok(path), Ok(root)) = (self.0.canonicalize(), std::env::temp_dir().canonicalize())
            else {
                return;
            };
            if path.parent() == Some(root.as_path())
                && path.file_name().is_some_and(|name| {
                    name.to_string_lossy()
                        .starts_with("rz-pals-checker-profile-test-")
                })
            {
                let _ = std::fs::remove_dir_all(path);
            }
        }
    }
    fn profile() -> PalsCheckerProfile {
        PalsCheckerProfile {
            schema: PALS_CHECKER_PROFILE_SCHEMA.into(),
            selection: PalsCheckerSelection::StockfishEmbeddedNnue,
            program: "/nonexistent-pals-profile-fixture/stockfish".into(),
            working_directory: "/nonexistent-pals-profile-fixture".into(),
            arguments: Vec::new(),
            identity: ExternalCheckerIdentity {
                adapter_semantics: EXTERNAL_UCI_SCORE_SEMANTICS.into(),
                binary_sha256: "a".repeat(64),
                launch_arguments_sha256: arguments_sha256(&[]),
                declared_name: "Stockfish".into(),
                declared_version: "fixture-declaration/1".into(),
                declared_source: STOCKFISH_SOURCE.into(),
                declared_license: "GPL-3.0-or-later".into(),
                options: BTreeMap::from([
                    ("Threads".into(), "2".into()),
                    ("Hash".into(), "16".into()),
                    ("Ponder".into(), "false".into()),
                    ("UCI_Chess960".into(), "false".into()),
                    ("MultiPV".into(), "1".into()),
                    ("SyzygyPath".into(), "".into()),
                    ("SyzygyProbeLimit".into(), "0".into()),
                ]),
                assets: Vec::new(),
                model_metadata: ExternalModelMetadata {
                    weights_sha256: None,
                    training: ExternalTrainingKnowledge::Unknown,
                    declared_rights: None,
                    precision: None,
                },
            },
            expected_uci_name: "Stockfish profile fixture".into(),
            max_depth: 8,
            max_prefix_plies: 64,
            handshake_timeout_ms: 1000,
            max_task_wall_time_ms: 1000,
            stop_grace_ms: 100,
            shutdown_grace_ms: 100,
            max_output_bytes: 4096,
            max_line_bytes: 1024,
        }
    }
    #[test]
    fn actual_file_pin_and_canonical_identity_are_distinct_and_no_spawn_occurs() {
        let root = OwnedTemp::new();
        let declared = profile();
        let (compact, compact_sha) =
            root.write("compact.json", &serde_json::to_vec(&declared).unwrap());
        let (pretty, pretty_sha) = root.write(
            "pretty.json",
            &serde_json::to_vec_pretty(&declared).unwrap(),
        );
        let first = PalsCheckerProfile::load(&compact, &compact_sha).unwrap();
        let second = PalsCheckerProfile::load(&pretty, &pretty_sha).unwrap();
        assert_ne!(first.file_sha256(), second.file_sha256());
        assert_eq!(first.canonical_sha256(), second.canonical_sha256());
        assert_eq!(first.profile(), &declared);
        let registration = first.registration();
        assert!(registration.declaration_only);
        assert!(!registration.process_start_performed);
        assert!(registration.observed_uci_name.is_none());
        assert!(registration.options_application_observed.is_none());
        assert!(registration.model_loading_observed.is_none());
        assert!(PalsCheckerProfile::load(&compact, &"b".repeat(64)).is_err());
        let mut changed = declared;
        changed
            .identity
            .options
            .insert("Threads".into(), "1".into());
        assert_ne!(
            changed.canonical_digest().unwrap(),
            first.canonical_sha256()
        );
        #[cfg(target_os = "linux")]
        {
            use rz_search::cpu_checker::CpuChecker;
            let checker = first.create_checker().unwrap();
            assert!(checker.last_attempt().is_none());
            assert_eq!(checker.diagnostic_output(), (&[][..], &[][..]));
            assert_eq!(checker.config().identity, first.profile().identity);
        }
        #[cfg(not(target_os = "linux"))]
        assert!(matches!(
            first.create_checker(),
            Err(CheckerError::Unsupported(_))
        ));
    }

    #[test]
    fn adaptive_bytes_have_distinct_schema_full_options_and_no_legacy_promotion() {
        // Fixed legacy digest was independently calculated from the documented
        // sorted JSON/domain recipe; the new admission cannot change old bytes.
        let legacy = profile();
        assert_eq!(
            legacy.canonical_digest().unwrap(),
            "1341ead9e0b1f5bc7c07b84e7c29e6ef082fa3af160ff06b4fb3bd3ce39fda9c"
        );
        let legacy_raw = serde_json::to_vec(&legacy).unwrap();
        let legacy_sha = format!("{:x}", Sha256::digest(&legacy_raw));
        assert!(
            PalsCheckerProfile::from_verified_adaptive_bytes(&legacy_raw, &legacy_sha).is_err()
        );
        let mut declared = legacy;
        declared.schema = PALS_ADAPTIVE_CHECKER_PROFILE_V5_SCHEMA.into();
        declared.identity.declared_name = "Stockfish 19".into();
        declared.identity.declared_version = "19".into();
        declared.identity.declared_source = format!(
            "{STOCKFISH_SOURCE}/tree/{}",
            rz_search::external_cpu::STOCKFISH19_SOURCE_COMMIT
        );
        declared.identity.options = BTreeMap::from([
            ("Skill Level".into(), "20".into()),
            ("UCI_LimitStrength".into(), "false".into()),
            ("Threads".into(), "1".into()),
            ("Hash".into(), "256".into()),
            ("Ponder".into(), "false".into()),
            ("NumaPolicy".into(), "none".into()),
            ("SyzygyPath".into(), String::new()),
            ("MultiPV".into(), "1".into()),
        ]);
        declared.expected_uci_name = "Stockfish 19".into();
        declared.stop_grace_ms = 500;
        let raw = serde_json::to_vec(&declared).unwrap();
        let digest = format!("{:x}", Sha256::digest(&raw));
        let loaded = PalsCheckerProfile::from_verified_adaptive_bytes(&raw, &digest).unwrap();
        assert_eq!(loaded.profile(), &declared);
        assert_eq!(loaded.file_bytes(), raw.len() as u64);
        assert_eq!(loaded.file_sha256(), digest);
        assert!(declared.validate().is_err());
        assert!(declared.canonical_digest().is_err());
        let expected = declared
            .canonical_digest_for(ADAPTIVE_CANONICAL_DOMAIN)
            .unwrap();
        assert_eq!(loaded.canonical_sha256(), expected);
        #[cfg(target_os = "linux")]
        {
            use rz_search::cpu_checker::CpuChecker;
            let config = loaded.adaptive_config().unwrap();
            assert_eq!(config.identity, declared.identity);
            assert_eq!(config.program, PathBuf::from(&declared.program));
            let checker = ExternalUciCpuChecker::create(config).unwrap();
            assert!(checker.last_attempt().is_none());
            assert_eq!(checker.diagnostic_output(), (&[][..], &[][..]));
        }
        #[cfg(not(target_os = "linux"))]
        assert!(matches!(
            loaded.adaptive_config(),
            Err(CheckerError::Unsupported(_))
        ));

        for change in 0..6 {
            let mut wrong = declared.clone();
            match change {
                0 => {
                    wrong.identity.options.insert("Threads".into(), "2".into());
                }
                1 => {
                    wrong
                        .identity
                        .options
                        .insert("UCI_Chess960".into(), "false".into());
                }
                2 => {
                    wrong.stop_grace_ms = 499;
                }
                3 => {
                    wrong.identity.declared_version = "18".into();
                }
                4 => {
                    wrong.identity.declared_source.push_str("/unrelated");
                }
                5 => {
                    wrong.arguments.push("--ignored".into());
                }
                _ => unreachable!(),
            }
            let raw = serde_json::to_vec(&wrong).unwrap();
            let digest = format!("{:x}", Sha256::digest(&raw));
            assert!(PalsCheckerProfile::from_verified_adaptive_bytes(&raw, &digest).is_err());
        }
        assert!(PalsCheckerProfile::from_verified_adaptive_bytes(&raw, &"b".repeat(64)).is_err());
        // A duplicate nested option must not silently replace its first value.
        let text = std::str::from_utf8(&raw)
            .unwrap()
            .replace("\"Threads\":\"1\"", "\"Threads\":\"1\",\"Threads\":\"1\"");
        let duplicate_sha = format!("{:x}", Sha256::digest(text.as_bytes()));
        assert!(
            PalsCheckerProfile::from_verified_adaptive_bytes(text.as_bytes(), &duplicate_sha)
                .is_err()
        );
    }
    #[test]
    fn profile_requires_bounded_bytes_regular_file_and_exact_wire_fields() {
        let root = OwnedTemp::new();
        let (path, digest) =
            root.write("oversized.json", &vec![b' '; MAX_CHECKER_PROFILE_BYTES + 1]);
        assert!(PalsCheckerProfile::load(&path, &digest).is_err());
        let (path, digest) = root.write("empty.json", &[]);
        assert!(PalsCheckerProfile::load(&path, &digest).is_err());
        assert!(PalsCheckerProfile::load(&root.0, &digest).is_err());
        let mut extra = serde_json::to_value(profile()).unwrap();
        extra["ignored"] = Value::Bool(true);
        let (path, digest) = root.write("unknown.json", &serde_json::to_vec(&extra).unwrap());
        assert!(PalsCheckerProfile::load(&path, &digest).is_err());
        extra.as_object_mut().unwrap().remove("ignored");
        extra["identity"]["ignored"] = Value::Bool(true);
        let (path, digest) =
            root.write("nested-unknown.json", &serde_json::to_vec(&extra).unwrap());
        assert!(PalsCheckerProfile::load(&path, &digest).is_err());
        let mut unsupported = serde_json::to_value(profile()).unwrap();
        unsupported["selection"] = Value::String("general_uci".into());
        let (path, digest) = root.write(
            "unsupported-selection.json",
            &serde_json::to_vec(&unsupported).unwrap(),
        );
        assert!(PalsCheckerProfile::load(&path, &digest).is_err());
        let mut padded = serde_json::to_vec(&profile()).unwrap();
        padded.resize(MAX_CHECKER_PROFILE_BYTES, b' ');
        let (path, digest) = root.write("exact-cap.json", &padded);
        assert!(PalsCheckerProfile::load(&path, &digest).is_ok());
        assert!(PalsCheckerProfile::load(Path::new("relative-profile.json"), &digest).is_err());
    }
    #[test]
    fn profile_rejects_duplicate_option_keys_before_canonical_registration() {
        let root = OwnedTemp::new();
        let original = serde_json::to_string(&profile()).unwrap();
        let key = r#""Threads":"2""#;
        assert!(original.contains(key));
        for (index, duplicate) in [
            r#""Threads":"1","Threads":"2""#,
            r#""Threads":"2","Threads":"2""#,
            r#""Threads":"1","\u0054hreads":"2""#,
        ]
        .into_iter()
        .enumerate()
        {
            let json = original.replacen(key, duplicate, 1);
            // The legacy typed BTreeMap alone accepts all these last-value
            // declarations. The file loader must reject their ambiguity.
            let typed: PalsCheckerProfile = serde_json::from_str(&json).unwrap();
            typed.validate().unwrap();
            assert_eq!(typed.identity.options["Threads"], "2");
            let (path, digest) = root.write(&format!("duplicate-{index}.json"), json.as_bytes());
            assert!(PalsCheckerProfile::load(&path, &digest).is_err());
        }
        assert!(
            serde_json::from_str::<UniqueJsonKeys>(r#"{"array":[{"key":1,"key":2}]}"#).is_err()
        );
        assert!(
            serde_json::from_str::<UniqueJsonKeys>(r#"{"array":[{"key":1},null,true,1.5,"text"]}"#)
                .is_ok()
        );
    }
    #[test]
    fn first_selection_rejects_argv_assets_unbounded_execution_and_unsafe_options() {
        let original = profile();
        original.validate().unwrap();
        for (name, value) in [
            ("Threads", "3"),
            ("Threads", "0"),
            ("Threads", "02"),
            ("Ponder", "true"),
            ("UCI_Chess960", "true"),
            ("MultiPV", "2"),
            ("SyzygyPath", "/tablebases"),
            ("SyzygyProbeLimit", "7"),
            ("Hash", "1025"),
            ("Hash", "0"),
            ("Hash", "016"),
            ("EvalFile", "network.nnue"),
        ] {
            let mut changed = original.clone();
            changed.identity.options.insert(name.into(), value.into());
            assert!(changed.validate().is_err(), "accepted {name}={value}");
        }
        for name in [
            "Threads",
            "Hash",
            "Ponder",
            "UCI_Chess960",
            "MultiPV",
            "SyzygyPath",
            "SyzygyProbeLimit",
        ] {
            let mut changed = original.clone();
            changed.identity.options.remove(name);
            assert!(changed.validate().is_err(), "accepted missing {name}");
        }
        let mut changed = original.clone();
        changed.arguments.push("bench".into());
        assert!(matches!(
            changed.validate(),
            Err(CheckerError::Unsupported(_))
        ));
        let mut changed = original.clone();
        changed
            .identity
            .assets
            .push(rz_search::cpu_checker::ExternalAssetIdentity {
                purpose: "nnue".into(),
                sha256: "b".repeat(64),
            });
        assert!(matches!(
            changed.validate(),
            Err(CheckerError::Unsupported(_))
        ));
        let mut changed = original.clone();
        changed.identity.model_metadata.weights_sha256 = Some("b".repeat(64));
        assert!(changed.validate().is_err());
        for field in 0..6 {
            let mut changed = original.clone();
            match field {
                0 => changed.identity.launch_arguments_sha256 = "b".repeat(64),
                1 => changed.identity.adapter_semantics = "own calibrated WDL".into(),
                2 => {
                    changed.identity.declared_source = "https://other-engine.example/source".into()
                }
                3 => changed.identity.declared_license = "MIT".into(),
                4 => changed.identity.declared_name = "Other Engine".into(),
                _ => changed.expected_uci_name = "Other Engine".into(),
            }
            assert!(changed.validate().is_err());
        }
        for path in [
            "stockfish",
            "C:/stockfish",
            "/bin/../stockfish",
            "/bin//stockfish",
            "/bin/stockfish\n",
        ] {
            let mut changed = original.clone();
            changed.program = path.into();
            assert!(changed.validate().is_err());
        }
        for duration in [0, 180_001] {
            for field in 0..4 {
                let mut changed = original.clone();
                match field {
                    0 => changed.handshake_timeout_ms = duration,
                    1 => changed.max_task_wall_time_ms = duration,
                    2 => changed.stop_grace_ms = duration,
                    _ => changed.shutdown_grace_ms = duration,
                }
                assert!(changed.validate().is_err());
            }
        }
        let mut changed = original.clone();
        changed.max_output_bytes = 16 * 1024 * 1024 + 1;
        assert!(changed.validate().is_err());
        let mut changed = original;
        changed.max_output_bytes = 1024 * 1024;
        changed.max_line_bytes = 65_537;
        assert!(changed.validate().is_err());
    }
    #[cfg(unix)]
    #[test]
    fn profile_rejects_file_and_parent_symlinks() {
        let root = OwnedTemp::new();
        let (path, digest) = root.write("profile.json", &serde_json::to_vec(&profile()).unwrap());
        let link = root.0.join("linked.json");
        std::os::unix::fs::symlink(&path, &link).unwrap();
        assert!(PalsCheckerProfile::load(&link, &digest).is_err());
        let parent = root.0.join("linked-parent");
        std::os::unix::fs::symlink(&root.0, &parent).unwrap();
        assert!(PalsCheckerProfile::load(&parent.join("profile.json"), &digest).is_err());
    }
}
