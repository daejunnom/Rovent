//! Read three exact, pinned CPU7 diagnostic inputs without issuing authority.
//!
//! Run this as a separate child of an owner that enforces a finite wall time,
//! CPU and memory limits, and child termination. The checks below are
//! cooperative: one regular-file I/O operation, hash, JSON decode, or Rules
//! operation cannot be preempted by this process's deadline callback. Input
//! bytes and returned observations do not authenticate a historical owner,
//! prove GPU completion, or authorize training.

#[cfg(target_os = "linux")]
fn main() -> std::process::ExitCode {
    match linux::run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("CPU7 diagnostic reader: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn main() -> std::process::ExitCode {
    eprintln!("CPU7 diagnostic reader requires Linux");
    std::process::ExitCode::FAILURE
}

#[cfg(target_os = "linux")]
mod linux {
    use cap_fs_ext::{DirExt, FollowSymlinks, OpenOptionsFollowExt};
    use cap_std::fs::{Dir, OpenOptions, OpenOptionsExt};
    use rz_experiments::rovent_utility_cohort::RoventCpu7CaptureRegistration;
    use rz_search::pals::store::{Cpu7DiagnosticExpected, StoreError, inspect_cpu7_diagnostic_ack};
    use sha2::{Digest, Sha256};
    use std::error::Error;
    use std::ffi::OsStr;
    use std::fs::File;
    use std::io::Read;
    use std::os::unix::fs::MetadataExt;
    use std::path::{Component, Path};
    use std::time::{Duration, Instant};

    type Result<T> = std::result::Result<T, Box<dyn Error>>;
    const MAX_META: u64 = 64 * 1024;
    const MAX_PROJECTION: u64 = 16 * 1024 * 1024;

    struct ReadBudget {
        deadline: Instant,
        limit: u64,
        charged: u64,
    }
    impl ReadBudget {
        fn check(&self) -> Result<()> {
            if Instant::now() >= self.deadline {
                return Err("original reader deadline expired".into());
            }
            Ok(())
        }
        fn reserve(&mut self, bytes: u64) -> Result<()> {
            self.check()?;
            self.charged = self
                .charged
                .checked_add(bytes)
                .filter(|value| *value <= self.limit)
                .ok_or("reader byte budget exhausted")?;
            Ok(())
        }
    }

    // The caller chooses only an explicit project root and relative artifact
    // names. Every opened directory component is checked without symlinks;
    // no broad search or recursive traversal is performed.
    fn forbidden(component: &str) -> bool {
        let s = component.to_ascii_lowercase();
        s == ".env"
            || s.starts_with(".env.")
            || s == ".ssh"
            || matches!(
                s.as_str(),
                ".aws"
                    | ".azure"
                    | ".kube"
                    | ".gnupg"
                    | ".docker"
                    | ".netrc"
                    | "gcloud"
                    | "auth.json"
                    | "config.toml"
            )
            || s.contains("credential")
            || s.contains("secret")
            || s.contains("service-account")
            || s.contains("service_account")
            || s.contains("serviceaccount")
            || s.contains("api-key")
            || s.contains("api_key")
            || s.contains("apikey")
            || s.contains("private_key")
            || s.contains("private-key")
            || s.contains("token")
            || s.contains("password")
            || s.starts_with("id_rsa")
            || s.starts_with("id_ed25519")
            || s.starts_with("id_ecdsa")
            || s.ends_with(".pem")
            || s.ends_with(".key")
            || s.ends_with(".p12")
            || s.ends_with(".pfx")
    }
    fn open_root(path: &Path) -> Result<Dir> {
        if !path.is_absolute() || path == Path::new("/") || path.as_os_str().len() > 4096 {
            return Err("an explicit non-root absolute project directory is required".into());
        }
        let mut root = Dir::open_ambient_dir("/", cap_std::ambient_authority())?;
        for part in path.components() {
            match part {
                Component::RootDir => {}
                Component::Normal(name) => {
                    let text = name
                        .to_str()
                        .ok_or("non-UTF8 project directory component")?;
                    if forbidden(text) {
                        return Err(
                            "project directory is outside the permitted artifact scope".into()
                        );
                    }
                    root = root.open_dir_nofollow(name)?;
                }
                _ => return Err("project directory must be normalized".into()),
            }
        }
        Ok(root)
    }
    fn relative_parts(value: &str) -> Result<Vec<&str>> {
        if value.is_empty()
            || value.len() > 4096
            || value.contains(['\\', ':'])
            || value.chars().any(char::is_control)
        {
            return Err("invalid relative artifact path".into());
        }
        let parts: Vec<_> = value.split('/').collect();
        if parts.iter().any(|part| {
            part.is_empty()
                || *part == "."
                || *part == ".."
                || part.starts_with('.')
                || forbidden(part)
        }) || Path::new(value).is_absolute()
        {
            return Err("invalid relative artifact path".into());
        }
        Ok(parts)
    }
    fn descend(root: &Dir, parts: &[&str]) -> Result<Dir> {
        let mut current = root.try_clone()?;
        for part in parts {
            current = current.open_dir_nofollow(OsStr::new(part))?;
        }
        Ok(current)
    }
    fn read_original(
        dir: &Dir,
        name: &str,
        max_bytes: u64,
        budget: &mut ReadBudget,
    ) -> Result<(File, Vec<u8>)> {
        budget.check()?;
        if relative_parts(name)?.len() != 1 {
            return Err("one exact artifact filename is required".into());
        }
        let mut options = OpenOptions::new();
        options
            .read(true)
            .follow(FollowSymlinks::No)
            .custom_flags(libc::O_CLOEXEC | libc::O_NONBLOCK);
        let mut file = dir.open_with(name, &options)?.into_std();
        let before = file.metadata()?;
        if !before.is_file() || before.len() == 0 || before.len() > max_bytes {
            return Err("artifact must be a bounded, nonempty regular file".into());
        }
        // Reserve the full content and one extra EOF-check byte, even when
        // EOF is observed or a later operation fails; no credit is refunded.
        budget.reserve(
            before
                .len()
                .checked_add(1)
                .ok_or("reader byte budget overflow")?,
        )?;
        let length = usize::try_from(before.len())?;
        let mut data = Vec::new();
        data.try_reserve_exact(length)?;
        let mut chunk = [0_u8; 8192];
        while data.len() < length {
            budget.check()?;
            let allowed = (length - data.len()).min(chunk.len());
            let got = file.read(&mut chunk[..allowed])?;
            if got == 0 {
                return Err("artifact ended before its original length".into());
            }
            data.extend_from_slice(&chunk[..got]);
        }
        budget.check()?;
        if file.read(&mut chunk[..1])? != 0 {
            return Err("artifact grew during its original read".into());
        }
        let after = file.metadata()?;
        if !after.is_file()
            || (
                after.dev(),
                after.ino(),
                after.len(),
                after.mtime(),
                after.mtime_nsec(),
                after.ctime(),
                after.ctime_nsec(),
            ) != (
                before.dev(),
                before.ino(),
                before.len(),
                before.mtime(),
                before.mtime_nsec(),
                before.ctime(),
                before.ctime_nsec(),
            )
        {
            return Err("artifact identity or metadata changed during read".into());
        }
        Ok((file, data))
    }
    fn sha256(value: &str) -> Result<[u8; 32]> {
        if value.len() != 64
            || !value
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err("64 lowercase hexadecimal SHA-256 required".into());
        }
        let mut result = [0_u8; 32];
        for (index, chunk) in value.as_bytes().chunks_exact(2).enumerate() {
            let digit = |b: u8| {
                if b.is_ascii_digit() {
                    b - b'0'
                } else {
                    b - b'a' + 10
                }
            };
            result[index] = (digit(chunk[0]) << 4) | digit(chunk[1]);
        }
        Ok(result)
    }

    pub fn run() -> Result<()> {
        let args: Vec<String> = std::env::args().skip(1).collect();
        if args.len() != 16 {
            return Err("usage: rovent_cpu7_diagnostic_reader INPUT_ROOT DIAGNOSTIC_ROOT REGISTRATION_REL ACK_SHA256 COHORT_ID TRIAL_SEQUENCE EPOCH GAME ROOT_GENERATION IMPLEMENTATION_SHA256 TRIAL_SHA256 READ_BYTES PROJECTION_ENTRIES DEPENDENCY_EDGES HISTORY_BYTES WALL_MS".into());
        }
        let registration = relative_parts(&args[2])?;
        let name = registration
            .last()
            .ok_or("registration filename required")?;
        let ack_sha = sha256(&args[3])?;
        sha256(&args[4])?;
        let sequence: u64 = args[5].parse()?;
        let epoch: u64 = args[6].parse()?;
        let game: u64 = args[7].parse()?;
        let root_generation: u64 = args[8].parse()?;
        sha256(&args[9])?;
        sha256(&args[10])?;
        let limit: u64 = args[11].parse()?;
        let entries: usize = args[12].parse()?;
        let edges: usize = args[13].parse()?;
        let history: usize = args[14].parse()?;
        let wall_ms: u64 = args[15].parse()?;
        if sequence == 0
            || limit == 0
            || wall_ms == 0
            || !(1..=65_536).contains(&entries)
            || edges > MAX_PROJECTION as usize
            || !(1..=64 * 1024 * 1024).contains(&history)
        {
            return Err("invalid trial, read, graph, history or wall limits".into());
        }
        let deadline = Instant::now()
            .checked_add(Duration::from_millis(wall_ms))
            .ok_or("original reader deadline overflow")?;
        let mut budget = ReadBudget {
            deadline,
            limit,
            charged: 0,
        };
        let input_root = open_root(Path::new(&args[0]))?;
        let diag_dir = open_root(Path::new(&args[1]))?;
        let reg_dir = descend(&input_root, &registration[..registration.len() - 1])?;
        let ack_name = format!("cpu7-selection-{sequence:06}-ack.json");
        let projection_name = format!("cpu7-selection-{sequence:06}.json");
        let (_ack_fd, ack) = read_original(&diag_dir, &ack_name, MAX_META, &mut budget)?;
        if <[u8; 32]>::from(Sha256::digest(&ack)) != ack_sha {
            return Err("original ACK SHA-256 differs from the caller pin".into());
        }
        let (_registration_fd, registration_raw) =
            read_original(&reg_dir, name, MAX_META, &mut budget)?;
        let text = std::str::from_utf8(&registration_raw)?;
        if !matches!(
            RoventCpu7CaptureRegistration::from_json(text),
            Ok(RoventCpu7CaptureRegistration::WithBundle(_))
        ) {
            return Err("explicit bounded CPU7 capture registration/2 required".into());
        }
        let (_projection_fd, projection) =
            read_original(&diag_dir, &projection_name, MAX_PROJECTION, &mut budget)?;
        let expected = Cpu7DiagnosticExpected {
            ack_sha256: ack_sha,
            cohort_id: &args[4],
            trial_sequence: sequence,
            process_epoch: epoch,
            game_generation: game,
            root_generation,
            implementation_sha256: &args[9],
            trial_semantic_sha256: &args[10],
            registration_path: &args[2],
            projection_path: &projection_name,
            reader_max_projection_entries: entries,
            reader_max_dependency_edges: edges,
            reader_max_history_charge_bytes: history,
        };
        let mut check = || {
            budget
                .check()
                .map_err(|_| StoreError::InvalidConditions("CPU7 reader deadline expired"))
        };
        let observation = inspect_cpu7_diagnostic_ack(
            &ack,
            &projection,
            &registration_raw,
            &expected,
            &mut check,
        )?;
        budget.check()?;
        if observation.training_authorized {
            return Err("historical inspection cannot authorize training".into());
        }
        println!(
            "{}",
            serde_json::json!({
                "schema":"rz-rovent-cpu7-diagnostic-reader-observation/1",
                "scope":"historical_only", "cohort_id":&args[4], "trial_sequence":sequence,
                "selected_records":observation.selected_records,
                "original_records":observation.original_records,
                "checked_cpu_observations":observation.checked_cpu_observations,
                "ack_content_join":observation.ack_content_join,
                "original_owner":observation.original_owner,
                "registration_and_ack":observation.registration_and_ack,
                "source_record_evidence_semantics":observation.source_record_evidence_semantics,
                "full_dependency_replay":observation.full_dependency_replay,
                "model_input_equivalence":observation.model_input_equivalence,
                "physical_completion":observation.physical_completion,
            "training_authorized":observation.training_authorized,
            "artifact_scope":"read_bytes_and_retained_file_descriptors_only",
            "files_rechecked_after_inspection":false,
                "read_budget":{"charged_bytes":budget.charged,"maximum_bytes":budget.limit,
                    "scope":"reserved_content_and_eof_checks"},
                "wall_enforcement":"cooperative_and_external_supervisor_required",
            })
        );
        Ok(())
    }
}
