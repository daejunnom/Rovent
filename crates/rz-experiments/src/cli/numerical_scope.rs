//! Read-only replay of the existing numerical gate. This caller opens and
//! retains original descriptors; it never creates a session, provider, startup
//! admission, native authority, optimizer or new physical-completion proof.

use std::ffi::OsString;
use std::path::PathBuf;

const MANAGED_ASSET_BYTES: u64 = 18 << 30;

#[derive(Debug)]
pub(super) struct Request {
    scope: PathBuf,
    scope_sha256: String,
    artifact_root: PathBuf,
    max_artifact_bytes: u64,
    timeout_ms: u64,
}

pub(super) fn parse(args: &[OsString]) -> Result<Request, String> {
    let scope = args
        .get(1)
        .filter(|path| !path.is_empty())
        .ok_or_else(|| "numerical scope path is required".to_string())?;
    let mut sha256 = None;
    let mut root = None;
    let mut bytes = None;
    let mut timeout = None;
    let mut index = 2;
    while index < args.len() {
        let value = args
            .get(index + 1)
            .ok_or_else(|| "numerical scope option requires a value".to_string())?;
        match args[index].to_str() {
            Some("--scope-sha256") if sha256.is_none() => {
                let value = value
                    .to_str()
                    .filter(|text| {
                        text.len() == 64
                            && text
                                .bytes()
                                .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
                    })
                    .ok_or_else(|| "--scope-sha256 requires lowercase SHA-256".to_string())?;
                sha256 = Some(value.to_string());
            }
            Some("--artifact-root") if root.is_none() && !value.is_empty() => {
                root = Some(PathBuf::from(value));
            }
            Some("--max-artifact-bytes") if bytes.is_none() => {
                let value = positive(value, "--max-artifact-bytes")?;
                if value > MANAGED_ASSET_BYTES {
                    return Err("artifact ceiling exceeds the 18GiB managed asset bound".into());
                }
                bytes = Some(value);
            }
            Some("--timeout-ms") if timeout.is_none() => {
                timeout = Some(positive(value, "--timeout-ms")?);
            }
            _ => return Err("unknown or duplicate numerical scope option".into()),
        }
        index += 2;
    }
    Ok(Request {
        scope: PathBuf::from(scope),
        scope_sha256: sha256.ok_or_else(|| "--scope-sha256 is required".to_string())?,
        artifact_root: root.ok_or_else(|| "--artifact-root is required".to_string())?,
        max_artifact_bytes: bytes.ok_or_else(|| "--max-artifact-bytes is required".to_string())?,
        timeout_ms: timeout.ok_or_else(|| "--timeout-ms is required".to_string())?,
    })
}

fn positive(value: &std::ffi::OsStr, option: &str) -> Result<u64, String> {
    value
        .to_str()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("{option} must be a positive u64"))
}

#[cfg(not(feature = "pals-content-admission"))]
pub(super) fn execute(request: Request) -> Result<String, String> {
    // Consume only the parsed arguments. An unsupported default build must not
    // open scope files, weights, runtime libraries or raw numerical captures.
    let _ = (
        request.scope,
        request.scope_sha256,
        request.artifact_root,
        request.max_artifact_bytes,
        request.timeout_ms,
    );
    Err("numerical scope verification is unsupported without the explicit pals-content-admission feature; no files or models were opened".into())
}

#[cfg(feature = "pals-content-admission")]
pub(super) fn execute(request: Request) -> Result<String, String> {
    supported::execute(request).map_err(|error| error.to_string())
}

#[cfg(feature = "pals-content-admission")]
mod supported {
    use super::Request;
    use cap_fs_ext::{FollowSymlinks, OpenOptionsFollowExt};
    use cap_std::fs::{Dir, OpenOptions};
    use rz_experiments::{
        ArtifactRef, MAX_MANIFEST_BYTES, ManifestError, PalsNumericalScopeV5,
        PalsStartupAssetPurposeV5, decode_json, validate_pals_numerical_scope_acceptance_v5,
    };
    use serde_json::json;
    use sha2::{Digest as _, Sha256};
    use std::collections::BTreeMap;
    use std::fs::File;
    use std::io::{Read as _, Seek as _, SeekFrom};
    use std::path::Path;
    use std::time::{Duration, Instant, SystemTime};

    const ACCEPTANCE_BYTES: usize = 256 << 10;

    #[derive(Debug, Eq, PartialEq)]
    struct FileStamp {
        bytes: u64,
        modified: Option<SystemTime>,
        created: Option<SystemTime>,
        #[cfg(unix)]
        unix_identity: (u64, u64, i64, i64),
    }
    impl FileStamp {
        fn read(file: &File) -> Result<Self, ManifestError> {
            let value = file
                .metadata()
                .map_err(|error| fail(format!("retained descriptor metadata: {error}")))?;
            if !value.is_file() {
                return Err(fail("retained descriptor is not a regular file"));
            }
            #[cfg(unix)]
            use std::os::unix::fs::MetadataExt as _;
            Ok(Self {
                bytes: value.len(),
                modified: value.modified().ok(),
                created: value.created().ok(),
                #[cfg(unix)]
                unix_identity: (value.dev(), value.ino(), value.ctime(), value.ctime_nsec()),
            })
        }
    }

    struct HeldArtifact {
        declaration: ArtifactRef,
        file: File,
        stamp: FileStamp,
    }
    impl HeldArtifact {
        fn read(&self, limit: usize, deadline: Instant) -> Result<Vec<u8>, ManifestError> {
            if self.declaration.bytes == 0 || self.declaration.bytes > limit as u64 {
                return Err(fail(
                    "requested numerical body exceeds its explicit read bound",
                ));
            }
            self.unchanged()?;
            let raw = read_bounded(&self.file, limit, deadline)?;
            self.unchanged()?;
            // The existing numerical validator verifies SHA and exact bytes
            // again on this very body; no numerical decision is duplicated.
            Ok(raw)
        }
        fn unchanged(&self) -> Result<(), ManifestError> {
            if FileStamp::read(&self.file)? != self.stamp {
                return Err(fail(
                    "retained artifact changed during numerical verification",
                ));
            }
            Ok(())
        }
    }

    fn fail(reason: impl Into<String>) -> ManifestError {
        ManifestError::Artifact(format!("numerical scope CLI: {}", reason.into()))
    }
    fn check_deadline(deadline: Instant) -> Result<(), ManifestError> {
        if Instant::now() >= deadline {
            return Err(fail("original numerical verification deadline expired"));
        }
        Ok(())
    }

    fn open_scope(path: &Path) -> Result<File, ManifestError> {
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let leaf = path
            .file_name()
            .ok_or_else(|| fail("scope requires a regular file leaf"))?;
        let root = Dir::open_ambient_dir(parent, cap_std::ambient_authority())
            .map_err(|error| fail(format!("scope parent directory: {error}")))?;
        let mut options = OpenOptions::new();
        options.read(true).follow(FollowSymlinks::No);
        #[cfg(unix)]
        {
            use cap_std::fs::OpenOptionsExt as _;
            options.custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW);
        }
        let file = root
            .open_with(leaf, &options)
            .map_err(|error| fail(format!("scope no-follow open: {error}")))?
            .into_std();
        let stamp = FileStamp::read(&file)?;
        if stamp.bytes == 0 || stamp.bytes > MAX_MANIFEST_BYTES as u64 {
            return Err(fail("scope JSON exceeds its 4MiB input bound"));
        }
        Ok(file)
    }

    fn read_bounded(
        file: &File,
        limit: usize,
        deadline: Instant,
    ) -> Result<Vec<u8>, ManifestError> {
        check_deadline(deadline)?;
        if limit == 0 || limit > MAX_MANIFEST_BYTES {
            return Err(fail("small numerical evidence read bound is invalid"));
        }
        let expected = FileStamp::read(file)?.bytes;
        if expected == 0 || expected > limit as u64 {
            return Err(fail("numerical evidence file exceeds the requested bound"));
        }
        let mut raw = Vec::new();
        raw.try_reserve_exact(expected as usize + 1)
            .map_err(|_| fail("bounded numerical evidence allocation failed"))?;
        let mut reader = file;
        reader
            .seek(SeekFrom::Start(0))
            .map_err(|error| fail(format!("retained numerical descriptor rewind: {error}")))?;
        let mut buffer = [0; 64 << 10];
        loop {
            check_deadline(deadline)?;
            let remaining = limit + 1 - raw.len();
            let requested = remaining.min(buffer.len());
            let read = reader
                .read(&mut buffer[..requested])
                .map_err(|error| fail(format!("retained numerical descriptor read: {error}")))?;
            if read == 0 {
                break;
            }
            raw.extend_from_slice(&buffer[..read]);
            if raw.len() > limit {
                return Err(fail("numerical evidence grew past its bounded read"));
            }
        }
        check_deadline(deadline)?;
        Ok(raw)
    }

    pub(super) fn execute(request: Request) -> Result<String, ManifestError> {
        let started = Instant::now();
        let deadline = started
            .checked_add(Duration::from_millis(request.timeout_ms))
            .ok_or_else(|| fail("explicit numerical verification deadline overflow"))?;
        let scope_file = open_scope(&request.scope)?;
        let scope_stamp = FileStamp::read(&scope_file)?;
        let scope_raw = read_bounded(&scope_file, MAX_MANIFEST_BYTES, deadline)?;
        if format!("{:x}", Sha256::digest(&scope_raw)) != request.scope_sha256 {
            return Err(fail("original scope bytes differ from --scope-sha256"));
        }
        let scope: PalsNumericalScopeV5 = decode_json(
            std::str::from_utf8(&scope_raw).map_err(|_| fail("scope JSON is not UTF-8"))?,
        )?;
        scope.validate_pre_acceptance()?;
        check_deadline(deadline)?;
        let total = scope.assets.iter().try_fold(0u64, |sum, asset| {
            sum.checked_add(asset.artifact.bytes)
                .ok_or_else(|| fail("declared asset byte sum overflow"))
        })?;
        if total > request.max_artifact_bytes {
            return Err(fail("declared assets exceed --max-artifact-bytes"));
        }
        let root = Dir::open_ambient_dir(&request.artifact_root, cap_std::ambient_authority())
            .map_err(|error| fail(format!("artifact root directory: {error}")))?;
        let mut held = BTreeMap::new();
        for asset in &scope.assets {
            check_deadline(deadline)?;
            // Reuse the original no-follow, streamed SHA/size verifier. That
            // existing call has no cancellation hook, so the deadline can be
            // checked before/after it, not enforced as hard preemption inside.
            let file = asset
                .artifact
                .open_verified_at(&root, request.max_artifact_bytes)?;
            check_deadline(deadline)?;
            let stamp = FileStamp::read(&file)?;
            held.insert(
                asset.artifact.path.clone(),
                HeldArtifact {
                    declaration: asset.artifact.clone(),
                    file,
                    stamp,
                },
            );
        }
        let read = |artifact: &ArtifactRef, limit: usize| -> Result<Vec<u8>, ManifestError> {
            check_deadline(deadline)?;
            let owner = held
                .get(&artifact.path)
                .filter(|owner| &owner.declaration == artifact)
                .ok_or_else(|| fail("numerical gate requested an unowned artifact"))?;
            owner.read(limit, deadline)
        };
        let acceptance = scope.asset(PalsStartupAssetPurposeV5::NumericalAcceptance)?;
        let acceptance_raw = read(&acceptance.artifact, ACCEPTANCE_BYTES)?;
        check_deadline(deadline)?;
        let admission = validate_pals_numerical_scope_acceptance_v5(&scope, &acceptance_raw, read)?;
        // The shared bounded comparator has no cancellation callback inside a
        // case. Reject an overrun on return; never label a late result accepted.
        check_deadline(deadline)?;
        for owner in held.values() {
            owner.unchanged()?;
        }
        if FileStamp::read(&scope_file)? != scope_stamp
            || read_bounded(&scope_file, MAX_MANIFEST_BYTES, deadline)? != scope_raw
        {
            return Err(fail("retained original scope changed during verification"));
        }
        check_deadline(deadline)?;
        // Diagnostic stdout only: preserve the existing admission verbatim
        // under a narrow envelope. This is not a new canonical persisted codec.
        let output = serde_json::to_string(&json!({
            "status":"saved_numerical_evidence_verified",
            "scope_sha256":request.scope_sha256,
            "numerical_admission":admission,
            "declared_artifact_bytes":total,
            "retained_artifact_descriptors":held.len(),
            "execution_ready":false,
            "native_authority_issued":false,
            "current_physical_completion_attested":false,
            "new_nn_execution_performed":false,
            "optimizer_updates_performed":0,
            "concurrent_inode_writes_frozen":false,
            "wall_clock_guard":{
                "requested_timeout_ms":request.timeout_ms,
                "elapsed_ms_before_serialization":started.elapsed().as_millis(),
                "hard_preemption":false,
                "scope":"cooperative_small_reads_and_boundaries_of_existing_streamed_hash_and_numerical_validator"
            }
        }))
        .map_err(|error| fail(format!("readonly result serialization: {error}")))?;
        check_deadline(deadline)?;
        Ok(output)
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::fs;
        use std::io::Write as _;
        use std::path::PathBuf;
        use std::sync::atomic::{AtomicU64, Ordering};

        static NEXT: AtomicU64 = AtomicU64::new(1);
        struct Fixture {
            root: PathBuf,
        }
        impl Fixture {
            fn new() -> Self {
                let root = std::env::temp_dir().join(format!(
                    "rz-numerical-cli-{}-{}",
                    std::process::id(),
                    NEXT.fetch_add(1, Ordering::Relaxed)
                ));
                fs::create_dir(&root).unwrap();
                Self { root }
            }
            fn artifact(&self, bytes: &[u8]) -> ArtifactRef {
                fs::write(self.root.join("evidence.json"), bytes).unwrap();
                ArtifactRef {
                    path: "evidence.json".into(),
                    sha256: format!("{:x}", Sha256::digest(bytes)),
                    bytes: bytes.len() as u64,
                    source: "https://example.org/authored-readonly-fd-fixture".into(),
                    license: "MIT".into(),
                }
            }
        }
        impl Drop for Fixture {
            fn drop(&mut self) {
                // Only these explicitly created leaves, never recursive cleanup.
                let _ = fs::remove_file(self.root.join("evidence.json"));
                let _ = fs::remove_file(self.root.join("original.json"));
                let _ = fs::remove_dir(&self.root);
            }
        }

        #[test]
        fn retained_evidence_detects_actual_inode_mutation() {
            let fixture = Fixture::new();
            let declaration = fixture.artifact(b"original");
            let file = declaration.open_verified(&fixture.root, 64).unwrap();
            let owner = HeldArtifact {
                stamp: FileStamp::read(&file).unwrap(),
                file,
                declaration,
            };
            let mut writer = fs::OpenOptions::new()
                .append(true)
                .open(fixture.root.join("evidence.json"))
                .unwrap();
            writer.write_all(b"changed").unwrap();
            writer.sync_all().unwrap();
            assert!(
                owner
                    .read(64, Instant::now() + Duration::from_secs(5))
                    .is_err()
            );
        }

        #[cfg(unix)]
        #[test]
        fn pathname_replacement_cannot_redirect_the_retained_evidence_fd() {
            let fixture = Fixture::new();
            let declaration = fixture.artifact(b"original");
            let file = declaration.open_verified(&fixture.root, 64).unwrap();
            let owner = HeldArtifact {
                stamp: FileStamp::read(&file).unwrap(),
                file,
                declaration: declaration.clone(),
            };
            fs::rename(
                fixture.root.join("evidence.json"),
                fixture.root.join("original.json"),
            )
            .unwrap();
            fs::write(fixture.root.join("evidence.json"), b"foreign!").unwrap();
            // The old FD still reads the original inode, while the substituted
            // pathname cannot pass the existing declared-artifact SHA gate.
            let raw =
                read_bounded(&owner.file, 64, Instant::now() + Duration::from_secs(5)).unwrap();
            assert_eq!(raw, b"original");
            assert!(declaration.open_verified(&fixture.root, 64).is_err());
        }

        #[test]
        fn expired_read_does_not_become_a_successful_empty_body() {
            let fixture = Fixture::new();
            let declaration = fixture.artifact(b"original");
            let file = declaration.open_verified(&fixture.root, 64).unwrap();
            assert!(read_bounded(&file, 64, Instant::now()).is_err());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_args() -> Vec<OsString> {
        [
            "pals-verify-numerical-scope",
            "scope.json",
            "--scope-sha256",
            &"a".repeat(64),
            "--artifact-root",
            "owned-artifacts",
            "--max-artifact-bytes",
            "4096",
            "--timeout-ms",
            "1000",
        ]
        .into_iter()
        .map(OsString::from)
        .collect()
    }

    #[test]
    fn scope_command_requires_all_explicit_pins_and_bounds() {
        let args = valid_args();
        let request = parse(&args).unwrap();
        assert_eq!(request.scope, PathBuf::from("scope.json"));
        assert_eq!(request.max_artifact_bytes, 4096);
        assert_eq!(request.timeout_ms, 1000);
        for offset in [2, 4, 6, 8] {
            let mut missing = args.clone();
            missing.drain(offset..offset + 2);
            assert!(parse(&missing).is_err());
        }
    }

    #[test]
    fn scope_command_rejects_duplicates_zero_and_foreign_arguments() {
        let args = valid_args();
        for (offset, value) in [(3, "A"), (7, "0"), (9, "0"), (8, "--unknown")] {
            let mut invalid = args.clone();
            invalid[offset] = value.into();
            assert!(parse(&invalid).is_err());
        }
        let mut duplicate = args;
        duplicate.extend(["--timeout-ms".into(), "1".into()]);
        assert!(parse(&duplicate).is_err());
    }

    #[cfg(not(feature = "pals-content-admission"))]
    #[test]
    fn default_build_is_explicitly_unsupported_without_opening_inputs() {
        let error = execute(parse(&valid_args()).unwrap()).unwrap_err();
        assert!(error.contains("unsupported"));
        assert!(error.contains("no files or models were opened"));
    }
}
