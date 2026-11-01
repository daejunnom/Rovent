//! Register an explicit PALS system-CUDA preflight from actual files.
//!
//! This program does not execute the target, load ORT, create a session, or
//! invoke a GPU. The caller owns the private output root and must prevent
//! concurrent writes, preserve any unknown owners, and supervise this process.
//! Usage: pals_system_registration <request.json> <private-root>
//!
//! The request supplies `config`, `binary`, `assets`, `system_runtime`,
//! `output`, and `max_seconds`. Each file has `snapshot_path`, `source`,
//! `license`, and a caller-selected `max_bytes`; an asset wraps this under
//! `file` and supplies its V5 `purpose`. All `inputs/...` parents and the
//! declared `baseline-runtime` or `candidate-runtime` directory must already
//! exist in the private root. The tool never copies any input or library.
//!
//! `config` contains the non-derived PALS V6 fields. It must omit `domain`,
//! `assets`, `system_runtime_lock`, `source_policy`, `source_components`,
//! `endpoint.binary`, `endpoint.source_commit`, `endpoint.model.weights`,
//! `endpoint.model.provenance`, and `endpoint.runtime.options`; these are
//! constructed from actual observed bytes and the runtime capture. Optional
//! `source_observation` may carry a caller-observed target inventory and its
//! binary SHA; it remains a claim until the child compares its compiled source.
//! When omitted, source identity is explicitly unknown. Model semantics and
//! execution/resource options are declarations, not automatically attested.
//! For `virtual_review_v1`, also omit `endpoint.model.semantic_sha256` and
//! `encoding_sha256`: these are computed from this registrar's compiled
//! implementation, which is not an assertion about a separate target binary.

#[cfg(not(all(target_os = "linux", feature = "pals-adaptive", feature = "onnx-cuda")))]
fn main() {
    eprintln!("registration requires Linux and features pals-adaptive,onnx-cuda");
    std::process::exit(2);
}

#[cfg(all(target_os = "linux", feature = "pals-adaptive", feature = "onnx-cuda"))]
mod registration {
    use cap_fs_ext::{DirExt, FollowSymlinks, OpenOptionsFollowExt};
    use cap_std::fs::{Dir, OpenOptions};
    use rz_eval::onnx::{SystemCaptureLimits, SystemLibraryRequest, SystemRuntime};
    use rz_experiments::{
        ArtifactRef, NativeRuntimeSourceV6, PALS_SYSTEM_STARTUP_CONFIG_V6_DOMAIN,
        PalsStartupAssetPurposeV5, PalsStartupAssetV5, PalsSystemNativeOptionsV6,
        PalsSystemSourcePolicyV6, PalsSystemStartupConfigV6, PalsSystemStartupLockV6,
        SystemLibrarySpecV6, SystemPlatformV6, SystemRuntimeLockV6, SystemRuntimeSpecV6,
        decode_json, pals_private_snapshot_path_v5,
    };
    use serde::Deserialize;
    use serde_json::{Map, Value};
    use sha2::{Digest, Sha256};
    use std::{
        collections::{BTreeMap, BTreeSet},
        fs::{self, File},
        io::{Read, Write},
        os::unix::fs::MetadataExt,
        path::{Component, Path, PathBuf},
        time::{Duration, Instant},
    };

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct FileInput {
        /// The actual private snapshot relative to the retained root.
        snapshot_path: String,
        /// Explicit provenance metadata; observing bytes does not verify it.
        source: String,
        license: String,
        /// Caller-specified per-file bound, not a compiled binary-size gate.
        max_bytes: u64,
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct AssetInput {
        purpose: PalsStartupAssetPurposeV5,
        file: FileInput,
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct LibraryInput {
        role: String,
        path: String,
        required_mapping: bool,
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct RuntimeInput {
        libraries: Vec<LibraryInput>,
        cuda_device_id: i32,
        cuda_arena_bytes: u64,
        startup_ms: u64,
        drain_ms: u64,
        capture_max_libraries: u32,
        capture_max_file_bytes: u64,
        capture_max_total_bytes: u64,
        capture_timeout_ms: u64,
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct SourceObservation {
        /// SHA of the binary from which the caller observed this inventory.
        binary_sha256: String,
        /// Explicit null means the compiled commit was not registered.
        #[serde(deserialize_with = "required_option")]
        source_commit: Option<String>,
        source_components: BTreeMap<String, String>,
    }
    fn required_option<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
    where
        D: serde::Deserializer<'de>,
        T: Deserialize<'de>,
    {
        Option::<T>::deserialize(deserializer)
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Request {
        /// V6 configuration without the derived fields documented below.
        config: Value,
        binary: FileInput,
        assets: Vec<AssetInput>,
        system_runtime: RuntimeInput,
        #[serde(default)]
        source_observation: Option<SourceObservation>,
        output: String,
        /// Overall registration window for file capture and publication.
        max_seconds: u64,
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }
    fn object<'a>(value: &'a mut Value, name: &str) -> Result<&'a mut Map<String, Value>, String> {
        value
            .as_object_mut()
            .ok_or_else(|| format!("{name} must be an object"))
    }
    fn insert_derived(map: &mut Map<String, Value>, key: &str, value: Value) -> Result<(), String> {
        if map.contains_key(key) {
            return Err(format!("{key} is derived and must not be supplied"));
        }
        map.insert(key.to_owned(), value);
        Ok(())
    }
    fn check_deadline(deadline: Instant) -> Result<(), String> {
        if Instant::now() >= deadline {
            Err("registration deadline expired".into())
        } else {
            Ok(())
        }
    }
    fn safe_external_path(path: &Path) -> bool {
        path.components().all(|component| {
            let text = component.as_os_str().to_string_lossy().to_ascii_lowercase();
            !(text == ".env"
                || text.starts_with(".env.")
                || text.contains("credential")
                || text.contains("secret")
                || text.contains("service-account")
                || text.contains("service_account")
                || text.contains("api-key")
                || text.contains("api_key")
                || text.contains("password")
                || text.contains("token")
                || text.starts_with("id_rsa")
                || text.starts_with("id_ed25519")
                || text == ".ssh")
        })
    }
    fn parent_and_name(root: &Dir, relative: &str) -> Result<(Dir, String), String> {
        if !pals_private_snapshot_path_v5(relative) {
            return Err("path must be a closed, non-secret inputs/ snapshot path".into());
        }
        let mut parts = relative.split('/').collect::<Vec<_>>();
        let name = parts.pop().ok_or("empty snapshot path")?.to_owned();
        let mut parent = root.try_clone().map_err(|e| format!("clone root: {e}"))?;
        for part in parts {
            parent = parent
                .open_dir_nofollow(part)
                .map_err(|e| format!("snapshot parent: {e}"))?;
        }
        Ok((parent, name))
    }
    /// Resolve an already-canonical absolute path without following any
    /// subsequently substituted parent symlink.
    fn absolute_parent(path: &Path) -> Result<(Dir, std::ffi::OsString), String> {
        let mut components = path.components();
        if !matches!(components.next(), Some(Component::RootDir)) {
            return Err("absolute path required".into());
        }
        let parts = components
            .map(|part| match part {
                Component::Normal(name) => Ok(name.to_os_string()),
                _ => Err("non-canonical path component".to_owned()),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (name, parents) = parts.split_last().ok_or("root path not allowed")?;
        let mut parent = Dir::open_ambient_dir("/", cap_std::ambient_authority())
            .map_err(|e| format!("system root: {e}"))?;
        for part in parents {
            parent = parent
                .open_dir_nofollow(part)
                .map_err(|e| format!("absolute parent: {e}"))?;
        }
        Ok((parent, name.clone()))
    }
    fn stamp(meta: &fs::Metadata) -> (u64, u64, u64, i64, i64, i64, i64, u64) {
        (
            meta.dev(),
            meta.ino(),
            meta.len(),
            meta.mtime(),
            meta.mtime_nsec(),
            meta.ctime(),
            meta.ctime_nsec(),
            meta.nlink(),
        )
    }
    fn verify_root(root: &Dir, path: &Path) -> Result<(), String> {
        let held = root
            .try_clone()
            .map_err(|e| format!("root clone: {e}"))?
            .into_std_file();
        let held = held.metadata().map_err(|e| format!("root fstat: {e}"))?;
        let named = fs::symlink_metadata(path).map_err(|e| format!("root restat: {e}"))?;
        if !held.is_dir()
            || !named.is_dir()
            || named.file_type().is_symlink()
            || held.dev() != named.dev()
            || held.ino() != named.ino()
        {
            return Err("private root was replaced during registration".into());
        }
        Ok(())
    }
    /// Returns the actual file identity and retains the original read-only FD.
    fn observe(
        root: &Dir,
        input: &FileInput,
        deadline: Instant,
    ) -> Result<(ArtifactRef, File), String> {
        check_deadline(deadline)?;
        if input.max_bytes == 0 {
            return Err("file bound must be positive".into());
        }
        let (parent, name) = parent_and_name(root, &input.snapshot_path)?;
        let named = parent
            .symlink_metadata(&name)
            .map_err(|e| format!("asset metadata: {e}"))?;
        if !named.is_file() {
            return Err("asset is not a regular file".into());
        }
        let mut options = OpenOptions::new();
        options.read(true).follow(FollowSymlinks::No);
        {
            use cap_std::fs::OpenOptionsExt;
            options.custom_flags(nix::libc::O_NONBLOCK | nix::libc::O_NOFOLLOW);
        }
        let mut file = parent
            .open_with(&name, &options)
            .map_err(|e| format!("asset open: {e}"))?
            .into_std();
        let before = file.metadata().map_err(|e| format!("asset fstat: {e}"))?;
        if !before.is_file() || before.len() == 0 || before.len() > input.max_bytes {
            return Err("asset is empty, non-regular or exceeds the declared bound".into());
        }
        let mut hash = Sha256::new();
        let mut remaining = before.len();
        let mut buf = [0u8; 64 * 1024];
        while remaining > 0 {
            check_deadline(deadline)?;
            let limit = remaining.min(buf.len() as u64) as usize;
            let n = file
                .read(&mut buf[..limit])
                .map_err(|e| format!("asset read: {e}"))?;
            if n == 0 {
                return Err("asset ended during observed read".into());
            }
            hash.update(&buf[..n]);
            remaining -= n as u64;
        }
        let after = file.metadata().map_err(|e| format!("asset fstat: {e}"))?;
        let named_after = parent
            .symlink_metadata(&name)
            .map_err(|e| format!("asset restat: {e}"))?;
        if stamp(&before) != stamp(&after)
            || cap_fs_ext::MetadataExt::dev(&named_after) != before.dev()
            || cap_fs_ext::MetadataExt::ino(&named_after) != before.ino()
            || !named_after.is_file()
        {
            return Err("asset changed or was replaced during registration".into());
        }
        let artifact = ArtifactRef {
            path: input.snapshot_path.clone(),
            sha256: hex(&hash.finalize()),
            bytes: before.len(),
            source: input.source.clone(),
            license: input.license.clone(),
        };
        artifact
            .validate()
            .map_err(|e| format!("artifact declaration: {e:?}"))?;
        Ok((artifact, file))
    }

    pub fn run() -> Result<(), String> {
        let mut args = std::env::args_os().skip(1);
        let request_path = PathBuf::from(args.next().ok_or("request path required")?);
        let root_path = PathBuf::from(args.next().ok_or("private root required")?);
        if args.next().is_some()
            || !request_path.is_absolute()
            || !root_path.is_absolute()
            || !safe_external_path(&request_path)
            || !safe_external_path(&root_path)
        {
            return Err("two absolute non-secret paths are required".into());
        }
        let request_real = fs::canonicalize(&request_path).map_err(|e| format!("request: {e}"))?;
        if !safe_external_path(&request_real) {
            return Err("secret-like resolved request path refused".into());
        }
        let meta = fs::symlink_metadata(&request_path).map_err(|e| format!("request: {e}"))?;
        if !meta.is_file() || meta.file_type().is_symlink() || meta.len() > 1024 * 1024 {
            return Err("request must be a bounded regular file".into());
        }
        let (request_parent, request_name) = absolute_parent(&request_real)?;
        let named_before = request_parent
            .symlink_metadata(&request_name)
            .map_err(|e| format!("resolved request metadata: {e}"))?;
        if !named_before.is_file() {
            return Err("resolved request is not a regular file".into());
        }
        let mut request_options = OpenOptions::new();
        use cap_std::fs::OpenOptionsExt as _;
        request_options
            .read(true)
            .follow(FollowSymlinks::No)
            .custom_flags(nix::libc::O_NOFOLLOW | nix::libc::O_NONBLOCK);
        let request_file = request_parent
            .open_with(&request_name, &request_options)
            .map_err(|e| format!("request open: {e}"))?;
        let request_file = request_file.into_std();
        let request_before = request_file
            .metadata()
            .map_err(|e| format!("request fstat: {e}"))?;
        if !request_before.is_file()
            || request_before.len() > 1024 * 1024
            || stamp(&request_before) != stamp(&meta)
            || cap_fs_ext::MetadataExt::dev(&named_before) != request_before.dev()
            || cap_fs_ext::MetadataExt::ino(&named_before) != request_before.ino()
        {
            return Err("request changed during open".into());
        }
        let mut input = String::new();
        (&request_file)
            .take(1024 * 1024 + 1)
            .read_to_string(&mut input)
            .map_err(|e| format!("request read: {e}"))?;
        let request_after = request_file
            .metadata()
            .map_err(|e| format!("request fstat: {e}"))?;
        let named_after = request_parent
            .symlink_metadata(&request_name)
            .map_err(|e| format!("resolved request restat: {e}"))?;
        if input.len() as u64 != request_before.len()
            || stamp(&request_before) != stamp(&request_after)
            || !named_after.is_file()
            || cap_fs_ext::MetadataExt::dev(&named_after) != request_before.dev()
            || cap_fs_ext::MetadataExt::ino(&named_after) != request_before.ino()
        {
            return Err("request changed during read".into());
        }
        let mut request: Request =
            decode_json(&input).map_err(|e| format!("request JSON: {e:?}"))?;
        if !(1..=3600).contains(&request.max_seconds) || request.assets.len() > 96 {
            return Err("invalid registration window or asset count".into());
        }
        let deadline = Instant::now()
            .checked_add(Duration::from_secs(request.max_seconds))
            .ok_or("registration deadline overflow")?;
        let root_meta =
            fs::symlink_metadata(&root_path).map_err(|e| format!("private root: {e}"))?;
        if !root_meta.is_dir() || root_meta.file_type().is_symlink() {
            return Err("private root must be an existing real directory".into());
        }
        let root_real = fs::canonicalize(&root_path).map_err(|e| format!("private root: {e}"))?;
        if root_real != root_path || !safe_external_path(&root_real) {
            return Err("private root must be a canonical non-secret directory".into());
        }
        for ancestor in root_path.ancestors() {
            match fs::symlink_metadata(ancestor.join(".git")) {
                Ok(_) => return Err("private root must be outside a source checkout".into()),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(format!("checkout boundary unavailable: {e}")),
            }
        }
        let (root_parent, root_name) = absolute_parent(&root_real)?;
        let root = root_parent
            .open_dir_nofollow(&root_name)
            .map_err(|e| format!("private root: {e}"))?;
        let held_meta = root
            .try_clone()
            .map_err(|e| format!("private root: {e}"))?
            .into_std_file()
            .metadata()
            .map_err(|e| format!("private root metadata: {e}"))?;
        if held_meta.dev() != root_meta.dev() || held_meta.ino() != root_meta.ino() {
            return Err("private root changed before it was pinned".into());
        }
        verify_root(&root, &root_path)?;
        let (binary, binary_fd) = observe(&root, &request.binary, deadline)?;
        if request
            .source_observation
            .as_ref()
            .is_some_and(|source| source.binary_sha256 != binary.sha256)
        {
            return Err("source observation refers to different binary bytes".into());
        }
        let mut files = vec![binary_fd];
        let mut paths = BTreeSet::new();
        paths.insert(request.binary.snapshot_path.clone());
        let mut weights = None;
        let mut provenance = None;
        let mut assets = Vec::new();
        for item in &request.assets {
            if !paths.insert(item.file.snapshot_path.clone()) {
                return Err("duplicate snapshot path".into());
            }
            let (artifact, fd) = observe(&root, &item.file, deadline)?;
            match item.purpose {
                PalsStartupAssetPurposeV5::Weights => {
                    if weights.replace(artifact.clone()).is_some() {
                        return Err("duplicate weights".into());
                    }
                }
                PalsStartupAssetPurposeV5::Provenance => {
                    if provenance.replace(artifact.clone()).is_some() {
                        return Err("duplicate provenance".into());
                    }
                }
                _ => {}
            }
            assets.push(PalsStartupAssetV5 {
                artifact,
                snapshot_path: item.file.snapshot_path.clone(),
                purpose: item.purpose,
            });
            files.push(fd);
        }
        let r = &request.system_runtime;
        if r.libraries
            .iter()
            .any(|library| !safe_external_path(Path::new(&library.path)))
        {
            return Err("secret-like system library path refused".into());
        }
        let mut requested = BTreeMap::new();
        let requests = r
            .libraries
            .iter()
            .map(|lib| {
                if requested
                    .insert(lib.role.clone(), lib.path.clone())
                    .is_some()
                {
                    return Err("duplicate library role".to_owned());
                }
                let resolved =
                    fs::canonicalize(&lib.path).map_err(|e| format!("system library path: {e}"))?;
                if !safe_external_path(&resolved) {
                    return Err("secret-like resolved library path refused".to_owned());
                }
                Ok(SystemLibraryRequest {
                    role: lib.role.clone(),
                    // Capture the already checked resolved path rather than
                    // following the supplied symlink a second time.
                    path: resolved,
                    required_mapping: lib.required_mapping,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let remaining = deadline
            .checked_duration_since(Instant::now())
            .ok_or("registration deadline expired")?;
        let capture_window = Duration::from_millis(r.capture_timeout_ms).min(remaining);
        let captured = SystemRuntime::capture(
            requests,
            SystemCaptureLimits {
                max_libraries: r.capture_max_libraries as usize,
                max_file_bytes: r.capture_max_file_bytes,
                max_total_bytes: r.capture_max_total_bytes,
                max_elapsed: capture_window,
            },
        )
        .map_err(|e| format!("system library capture: {e:?}"))?;
        let libraries = captured
            .libraries()
            .iter()
            .map(|id| -> Result<SystemLibrarySpecV6, String> {
                Ok(SystemLibrarySpecV6 {
                    role: id.role.clone(),
                    requested_path: requested
                        .get(&id.role)
                        .ok_or("unexpected library role")?
                        .clone(),
                    canonical_path: id
                        .canonical_path
                        .to_str()
                        .ok_or("library path is not UTF-8")?
                        .to_owned(),
                    version: None,
                    sha256: hex(&id.sha256),
                    bytes: id.bytes,
                    device: id.device,
                    inode: id.inode,
                    required_mapping: id.required_mapping,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let spec = SystemRuntimeSpecV6 {
            platform: SystemPlatformV6::Linux,
            libraries,
            system_identity_sha256: hex(&captured.identity_sha256()),
            cuda_device_id: r.cuda_device_id,
            cuda_arena_bytes: r.cuda_arena_bytes,
            startup_ms: r.startup_ms,
            drain_ms: r.drain_ms,
            capture_max_libraries: r.capture_max_libraries,
            capture_max_file_bytes: r.capture_max_file_bytes,
            capture_max_total_bytes: r.capture_max_total_bytes,
            capture_timeout_ms: r.capture_timeout_ms,
        };
        let runtime_lock =
            SystemRuntimeLockV6::new(NativeRuntimeSourceV6::SystemCuda(spec.clone()))
                .map_err(|e| format!("system runtime lock: {e:?}"))?;
        let config = object(&mut request.config, "config")?;
        insert_derived(
            config,
            "domain",
            Value::String(PALS_SYSTEM_STARTUP_CONFIG_V6_DOMAIN.into()),
        )?;
        insert_derived(
            config,
            "assets",
            serde_json::to_value(assets).map_err(|e| e.to_string())?,
        )?;
        insert_derived(
            config,
            "system_runtime_lock",
            serde_json::to_value(runtime_lock).map_err(|e| e.to_string())?,
        )?;
        // This tool never executes the target binary. Caller-observed source
        // information is a claim until the child checks its compiled values.
        insert_derived(
            config,
            "source_policy",
            serde_json::to_value(PalsSystemSourcePolicyV6::Observe).map_err(|e| e.to_string())?,
        )?;
        let (source_commit, source_components) = match request.source_observation.take() {
            Some(source) => (source.source_commit, source.source_components),
            None => (None, BTreeMap::new()),
        };
        insert_derived(
            config,
            "source_components",
            serde_json::to_value(source_components).map_err(|e| e.to_string())?,
        )?;
        let native_options: PalsSystemNativeOptionsV6 = serde_json::from_value(
            config
                .get("native_options")
                .cloned()
                .ok_or("native_options required")?,
        )
        .map_err(|e| format!("native_options: {e}"))?;
        let endpoint = object(
            config.get_mut("endpoint").ok_or("endpoint required")?,
            "endpoint",
        )?;
        insert_derived(
            endpoint,
            "binary",
            serde_json::to_value(binary).map_err(|e| e.to_string())?,
        )?;
        insert_derived(
            endpoint,
            "source_commit",
            serde_json::to_value(source_commit).map_err(|e| e.to_string())?,
        )?;
        let model = object(endpoint.get_mut("model").ok_or("model required")?, "model")?;
        let semantics_scope =
            if model.get("profile").and_then(Value::as_str) == Some("virtual_review_v1") {
                let semantic = rz_eval::pals_model::PalsModelProfile::VirtualReviewV1
                    .registered_semantics_sha256();
                let encoding = rz_eval::pals_model::virtual_review::registered_encoding_sha256();
                insert_derived(model, "semantic_sha256", Value::String(hex(&semantic)))?;
                insert_derived(model, "encoding_sha256", Value::String(hex(&encoding)))?;
                "registrar_compiled_only_target_unverified"
            } else {
                "caller_declared_unverified"
            };
        insert_derived(
            model,
            "weights",
            serde_json::to_value(weights.ok_or("weights asset required")?)
                .map_err(|e| e.to_string())?,
        )?;
        insert_derived(
            model,
            "provenance",
            serde_json::to_value(provenance).map_err(|e| e.to_string())?,
        )?;
        let runtime = object(
            endpoint
                .get_mut("runtime")
                .ok_or("runtime component required")?,
            "runtime",
        )?;
        insert_derived(
            runtime,
            "options",
            serde_json::to_value(native_options.canonical_runtime_options(&spec))
                .map_err(|e| e.to_string())?,
        )?;
        let encoded = serde_json::to_string(&request.config).map_err(|e| format!("config: {e}"))?;
        let typed = PalsSystemStartupConfigV6::from_json(&encoded)
            .map_err(|e| format!("config validation: {e:?}"))?;
        let _receipt_parent = root
            .open_dir_nofollow(&typed.runtime_directory)
            .map_err(|e| format!("preflight receipt directory unavailable: {e}"))?;
        let lock = PalsSystemStartupLockV6::new(typed).map_err(|e| format!("lock: {e:?}"))?;
        let output = lock
            .canonical_bytes()
            .map_err(|e| format!("lock encoding: {e:?}"))?;
        check_deadline(deadline)?;
        verify_root(&root, &root_path)?;
        let (parent, name) = parent_and_name(&root, &request.output)?;
        let mut options = OpenOptions::new();
        options
            .write(true)
            .create_new(true)
            .follow(FollowSymlinks::No);
        let mut out = parent
            .open_with(&name, &options)
            .map_err(|e| format!("lock create: {e}"))?
            .into_std();
        out.write_all(&output)
            .map_err(|e| format!("lock write: {e}"))?;
        out.sync_all().map_err(|e| format!("lock sync: {e}"))?;
        verify_root(&root, &root_path)?;
        // FDs and the system capture remain alive until publication has ended.
        drop(files);
        drop(captured);
        println!(
            "{}",
            serde_json::json!({
                "status":"registered", "output":request.output, "canonical_sha256":lock.canonical_sha256,
                "model_and_binary_bytes_observed":true, "declared_source_verified":false,
                "model_semantics_observation":semantics_scope,
                "target_binary_executed":false, "provider_ready":false, "gpu_inference":false
            })
        );
        Ok(())
    }
}

#[cfg(all(target_os = "linux", feature = "pals-adaptive", feature = "onnx-cuda"))]
fn main() {
    if let Err(error) = registration::run() {
        eprintln!("PALS system registration failed: {error}");
        std::process::exit(1);
    }
}
