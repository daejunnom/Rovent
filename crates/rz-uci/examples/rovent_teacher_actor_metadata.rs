//! A prospective teacher actor from one original pinned profile declaration.
//!
//! `--profile=ABS.json --profile-sha256=SHA --profile-bytes=N --actor-id=ID`
//! Referenced executables/assets are never opened or started. The exact twelve
//! fields are metadata for a new plan, not a historical actor or any authority.

#![forbid(unsafe_code)]

#[cfg(not(feature = "search-work-receipts"))]
fn main() -> std::process::ExitCode {
    use std::io::Write as _;
    let _ = std::io::stderr().lock().write_all(
        b"{\"schema\":\"rz-rovent-teacher-actor-metadata-error/1\",\"code\":\"requires_search_work_receipts\",\"execution_authorized\":false,\"performed_optimizer_updates\":0}\n",
    );
    std::process::ExitCode::from(2)
}

#[cfg(feature = "search-work-receipts")]
fn main() -> std::process::ExitCode {
    metadata::main()
}

#[cfg(feature = "search-work-receipts")]
mod metadata {
    use rz_uci::pals_checker_profile::MAX_CHECKER_PROFILE_BYTES;
    use rz_uci::pals_checker_profile::rovent::RoventExternalProfile;
    use std::ffi::OsString;
    use std::fs::{File, symlink_metadata};
    use std::io::{Read, Write};
    use std::path::PathBuf;
    use std::process::ExitCode;

    struct Arguments {
        profile: PathBuf,
        sha256: String,
        bytes: u64,
        actor_id: String,
    }

    fn arguments(args: impl IntoIterator<Item = OsString>) -> Result<Arguments, &'static str> {
        let (mut profile, mut sha256, mut bytes, mut actor_id) = (None, None, None, None);
        for arg in args {
            let arg = arg.to_str().ok_or("non_utf8_argument")?;
            if let Some(value) = arg.strip_prefix("--profile=") {
                if profile.is_some() {
                    return Err("duplicate_profile");
                }
                let path = PathBuf::from(value);
                let name = path
                    .file_name()
                    .and_then(|v| v.to_str())
                    .ok_or("profile_path")?;
                let lower = name.to_ascii_lowercase();
                if !path.is_absolute()
                    || path.extension().and_then(|v| v.to_str()) != Some("json")
                    || lower.starts_with('.')
                    || [
                        "credential",
                        "secret",
                        "service-account",
                        "service_account",
                        "private-key",
                        "private_key",
                        "api-key",
                        "api_key",
                    ]
                    .iter()
                    .any(|part| lower.contains(part))
                {
                    return Err("explicit_nonsecret_profile_json_required");
                }
                profile = Some(path);
            } else if let Some(value) = arg.strip_prefix("--profile-sha256=") {
                if sha256.is_some()
                    || value.len() != 64
                    || !value
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                {
                    return Err("profile_sha256");
                }
                sha256 = Some(value.to_owned());
            } else if let Some(value) = arg.strip_prefix("--profile-bytes=") {
                if bytes.is_some() {
                    return Err("duplicate_profile_bytes");
                }
                let size = value.parse::<u64>().map_err(|_| "profile_byte_count")?;
                if size == 0 || size > MAX_CHECKER_PROFILE_BYTES as u64 {
                    return Err("profile_byte_count");
                }
                bytes = Some(size);
            } else if let Some(value) = arg.strip_prefix("--actor-id=") {
                if actor_id.is_some() || value.is_empty() || value.len() > 256 {
                    return Err("actor_id");
                }
                actor_id = Some(value.to_owned());
            } else {
                return Err("unexpected_argument");
            }
        }
        Ok(Arguments {
            profile: profile.ok_or("missing_profile")?,
            sha256: sha256.ok_or("missing_profile_sha256")?,
            bytes: bytes.ok_or("missing_profile_bytes")?,
            actor_id: actor_id.ok_or("missing_actor_id")?,
        })
    }

    fn run() -> Result<(), &'static str> {
        let args = arguments(std::env::args_os().skip(1))?;
        let before = symlink_metadata(&args.profile).map_err(|_| "profile_metadata")?;
        if !before.file_type().is_file()
            || before.file_type().is_symlink()
            || before.len() != args.bytes
        {
            return Err("original_regular_profile_required");
        }
        let input = File::open(&args.profile).map_err(|_| "profile_open")?;
        let opened = input.metadata().map_err(|_| "opened_profile_metadata")?;
        if !opened.is_file() || opened.len() != args.bytes {
            return Err("opened_profile_byte_count");
        }
        let mut raw = Vec::with_capacity(args.bytes as usize);
        input
            .take(args.bytes + 1)
            .read_to_end(&mut raw)
            .map_err(|_| "profile_read")?;
        if raw.len() as u64 != args.bytes {
            return Err("original_profile_byte_count");
        }
        let loaded = RoventExternalProfile::from_verified_bytes(&raw, &args.sha256)
            .map_err(|_| "original_profile_rejected")?;
        let actor = loaded
            .prospective_teacher_actor(&args.actor_id)
            .map_err(|_| "prospective_teacher_declaration_rejected")?;
        let mut output = serde_json::to_vec(&actor).map_err(|_| "metadata_encoding")?;
        if output.len() >= MAX_CHECKER_PROFILE_BYTES {
            return Err("metadata_output_budget");
        }
        output.push(b'\n');
        std::io::stdout()
            .lock()
            .write_all(&output)
            .map_err(|_| "metadata_stdout")
    }

    pub fn main() -> ExitCode {
        match run() {
            Ok(()) => ExitCode::SUCCESS,
            Err(code) => {
                let output = serde_json::json!({
                    "schema":"rz-rovent-teacher-actor-metadata-error/1",
                    "code":code,
                    "execution_authorized":false,
                    "performed_optimizer_updates":0
                });
                let _ = writeln!(std::io::stderr().lock(), "{output}");
                ExitCode::from(2)
            }
        }
    }
}
