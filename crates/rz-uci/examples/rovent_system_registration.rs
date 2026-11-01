//! Create new system-CUDA Rovent registrations from explicit current inputs.
//!
//! This does not launch the engine, load ONNX Runtime or perform GPU inference.
//! It captures current native files and checks typed declarations. Target binary,
//! model and provenance bytes still require their normal admission verification.
//! Run on Linux with: cargo run -p rz-uci --example rovent_system_registration
//! --features pals-adaptive,onnx-cuda -- <request.json> <existing-output-root>

#[cfg(not(all(target_os = "linux", feature = "pals-adaptive", feature = "onnx-cuda")))]
fn main() {
    eprintln!("registration requires Linux and features pals-adaptive,onnx-cuda");
    std::process::exit(2);
}

#[cfg(all(target_os = "linux", feature = "pals-adaptive", feature = "onnx-cuda"))]
mod registration {
    use rz_eval::onnx::{SystemCaptureLimits, SystemLibraryRequest, SystemRuntime};
    use rz_experiments::{
        ArtifactRef, NativeRuntimeSourceV6, ROVENT_EXECUTION_V3_DOMAIN,
        ROVENT_STARTUP_DOCUMENT_V5_DOMAIN, RoventCoordinatorDeclaration, RoventEngineDeclaration,
        RoventExecutionManifestV3, RoventPurpose, RoventResourcePlan, RoventRootTieDeclarationV2,
        RoventStartupDocumentV5, SystemLibrarySpecV6, SystemPlatformV6, SystemRuntimeLockV6,
        SystemRuntimeSpecV6,
    };
    use rz_uci::rovent_bootstrap::{
        configuration::v5::{RoventStartupConfigurationV5, RoventTypedStartupV5},
        process_boundary::compiled_process_source_components_v3,
        system_configuration_digest_v3,
    };
    use serde::Deserialize;
    use sha2::{Digest, Sha256};
    use std::{
        collections::HashMap,
        fs::{self, File, OpenOptions},
        io::{Read, Write},
        os::unix::fs::MetadataExt,
        path::{Component, Path, PathBuf},
        time::Duration,
    };

    const REQUEST_MAX: u64 = 1 << 20;

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Library {
        role: String,
        path: String,
        required_mapping: bool,
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct RuntimeRequest {
        libraries: Vec<Library>,
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
    struct Outputs {
        startup: String,
        runtime_lock: String,
        manifest: String,
        manifest_canonical: String,
        input_lock: String,
        startup_source: String,
        startup_license: String,
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Request {
        contract_revision: String,
        run_id: String,
        purpose: RoventPurpose,
        engine: RoventEngineDeclaration,
        resources: RoventResourcePlan,
        coordinator: RoventCoordinatorDeclaration,
        wall_time_max_ms: u64,
        cleanup_max_ms: u64,
        output_bytes_max: u64,
        agent_renewable: bool,
        provenance: Vec<ArtifactRef>,
        own_r_root_tie: RoventRootTieDeclarationV2,
        startup_configuration: RoventStartupConfigurationV5,
        system_runtime: RuntimeRequest,
        outputs: Outputs,
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }
    fn output_path(root: &Path, name: &str) -> Result<PathBuf, String> {
        if !rz_experiments::pals_private_snapshot_path_v5(name) {
            return Err("output must use the closed inputs/ snapshot namespace".into());
        }
        let relative = Path::new(name);
        let mut parent = root.to_path_buf();
        let mut components = relative.components().peekable();
        while let Some(part) = components.next() {
            let Component::Normal(name) = part else {
                return Err("invalid output path".into());
            };
            parent.push(name);
            if components.peek().is_some() {
                let meta =
                    fs::symlink_metadata(&parent).map_err(|e| format!("output parent: {e}"))?;
                if !meta.is_dir() || meta.file_type().is_symlink() {
                    return Err("output parents must be existing real directories".into());
                }
            }
        }
        if !parent.starts_with(root) {
            return Err("output escapes root".into());
        }
        Ok(parent)
    }
    fn request(path: &Path) -> Result<Request, String> {
        let meta = fs::symlink_metadata(path).map_err(|e| format!("request metadata: {e}"))?;
        if !meta.is_file() || meta.file_type().is_symlink() || meta.len() > REQUEST_MAX {
            return Err("request must be a regular file of at most 1 MiB".into());
        }
        if path.components().any(|part| match part {
            Component::Normal(name) => {
                let name = name.to_string_lossy().to_ascii_lowercase();
                name == ".env"
                    || name.starts_with(".env.")
                    || name.contains("credential")
                    || name.contains("secret")
            }
            _ => false,
        }) {
            return Err("secret-like request paths are not allowed".into());
        }
        let mut bytes = Vec::new();
        File::open(path)
            .map_err(|e| format!("request open: {e}"))?
            .take(REQUEST_MAX + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| format!("request read: {e}"))?;
        if bytes.len() as u64 > REQUEST_MAX {
            return Err("request exceeds 1 MiB".into());
        }
        serde_json::from_slice(&bytes).map_err(|e| format!("request JSON: {e}"))
    }

    pub fn run() -> Result<(), String> {
        let args: Vec<_> = std::env::args_os().collect();
        if args.len() != 3 {
            return Err(
                "usage: rovent_system_registration <request.json> <existing-output-root>".into(),
            );
        }
        let mut req = request(Path::new(&args[1]))?;
        let output = Path::new(&args[2]);
        if !output.is_absolute() {
            return Err("output root must be absolute".into());
        }
        let meta = fs::symlink_metadata(output).map_err(|e| format!("output root: {e}"))?;
        if !meta.is_dir() || meta.file_type().is_symlink() {
            return Err("output root must be an existing real directory".into());
        }
        let root =
            fs::canonicalize(output).map_err(|e| format!("output root canonicalization: {e}"))?;
        let names = [
            &req.outputs.startup,
            &req.outputs.runtime_lock,
            &req.outputs.manifest,
            &req.outputs.manifest_canonical,
            &req.outputs.input_lock,
        ];
        let mut unique = std::collections::HashSet::new();
        for name in names {
            let path = output_path(&root, name)?;
            if !unique.insert(name) {
                return Err("output paths must be distinct".into());
            }
            if fs::symlink_metadata(path).is_ok() {
                return Err(format!("output already exists: {name}"));
            }
        }
        let r = &req.system_runtime;
        if r.capture_max_libraries == 0
            || r.capture_max_file_bytes == 0
            || r.capture_max_total_bytes == 0
            || r.capture_timeout_ms == 0
        {
            return Err("capture budgets must be nonzero".into());
        }
        let actual_components = compiled_process_source_components_v3()
            .map_err(|e| format!("compiled source inventory: {e:?}"))?;
        let mut required = vec![
            "rules",
            "store",
            "review_view",
            "coordinator",
            "role_tickets",
            "adaptive_costs",
            "resource_boundary",
            "uci",
            "model_native_v3",
            "model_encoding",
            "nn_runtime",
            "rovent_system_source_v3",
            "rovent_process_boundary",
        ];
        match req.engine.researcher.as_ref() {
            Some(rz_experiments::RoventResearcher::Own { .. }) => {
                required.extend(["cpu_search", "cpu_worker", "cpu_credit"])
            }
            Some(rz_experiments::RoventResearcher::External { .. }) => {
                required.push("external_cpu")
            }
            None => {}
        }
        if req.engine.nnuec.is_some() && !required.contains(&"cpu_search") {
            required.push("cpu_search");
        }
        if required
            .iter()
            .any(|name| !req.engine.source_components.contains_key(*name))
            || req.engine.source_components.len() > 64
            || req
                .engine
                .source_components
                .iter()
                .any(|(k, v)| actual_components.get(k) != Some(v))
        {
            return Err("required source pins missing or different from registration executable's embedded sources".into());
        }
        if req.engine.source_components.get("model_native_v3")
            != Some(&req.engine.pc.proposer.implementation_sha256)
        {
            return Err(
                "model native source pin differs from explicitly registered model implementation"
                    .into(),
            );
        }
        let requests: Vec<SystemLibraryRequest> = r
            .libraries
            .iter()
            .map(|l| SystemLibraryRequest {
                role: l.role.clone(),
                path: PathBuf::from(&l.path),
                required_mapping: l.required_mapping,
            })
            .collect();
        let captured = SystemRuntime::capture(
            requests,
            SystemCaptureLimits {
                max_libraries: r.capture_max_libraries as usize,
                max_file_bytes: r.capture_max_file_bytes,
                max_total_bytes: r.capture_max_total_bytes,
                max_elapsed: Duration::from_millis(r.capture_timeout_ms),
            },
        )
        .map_err(|e| format!("system library capture failed: {e:?}"))?;
        let mut requested = HashMap::new();
        for lib in &r.libraries {
            if requested
                .insert(lib.role.clone(), lib.path.clone())
                .is_some()
            {
                return Err("duplicate system library role".into());
            }
        }
        let identities = captured.libraries();
        let core = identities
            .iter()
            .find(|l| l.role == "ort-core")
            .ok_or("captured ort-core missing")?;
        let libraries = identities
            .iter()
            .map(|id| -> Result<SystemLibrarySpecV6, String> {
                let requested_path = requested
                    .get(&id.role)
                    .ok_or("captured unexpected role")?
                    .clone();
                let canonical_path = id
                    .canonical_path
                    .to_str()
                    .ok_or("system path is not UTF-8")?
                    .to_owned();
                Ok(SystemLibrarySpecV6 {
                    role: id.role.clone(),
                    requested_path,
                    canonical_path,
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
        let sys_lock = SystemRuntimeLockV6::new(NativeRuntimeSourceV6::SystemCuda(spec.clone()))
            .map_err(|e| format!("system runtime registration: {e:?}"))?;
        let doc = RoventStartupDocumentV5 {
            domain: ROVENT_STARTUP_DOCUMENT_V5_DOMAIN.into(),
            system_runtime_lock_sha256: sys_lock.canonical_sha256.clone(),
            configuration: serde_json::to_value(&req.startup_configuration)
                .map_err(|e| format!("startup serialization: {e}"))?,
        };
        let mut startup_value =
            serde_json::to_value(&doc).map_err(|e| format!("startup envelope: {e}"))?;
        startup_value.sort_all_objects();
        let startup =
            serde_json::to_vec(&startup_value).map_err(|e| format!("startup envelope: {e}"))?;
        let typed = RoventTypedStartupV5::from_json(&startup)
            .map_err(|e| format!("typed startup: {e:?}"))?;
        typed
            .require_supported_native_capabilities()
            .map_err(|e| format!("native capability declaration: {e:?}"))?;
        let onnx = typed
            .fields()
            .native
            .onnx_system(&req.engine.pc.device, &spec)
            .map_err(|e| format!("native configuration and system selection disagree: {e:?}"))?;
        // These values are derived from actual captured native bytes and the typed
        // startup, never carried over from an earlier run or caller placeholders.
        req.engine.pc.backend_configuration_sha256 =
            hex(
                &system_configuration_digest_v3(onnx, core.sha256, captured.identity_sha256())
                    .map_err(|e| format!("native configuration identity: {e:?}"))?,
            );
        req.engine.search.search_configuration_sha256 =
            hex(&typed
                .search_configuration_digest()
                .map_err(|e| format!("search identity: {e:?}"))?);
        // Policy identity is the digest of this exact typed startup configuration,
        // computed at registration time rather than copied from any fixture or run.
        let startup_configuration_sha256 = hex(&typed
            .configuration_digest()
            .map_err(|e| format!("startup identity: {e:?}"))?);
        req.coordinator.policy_sha256 = startup_configuration_sha256.clone();
        let manifest = RoventExecutionManifestV3 {
            domain: ROVENT_EXECUTION_V3_DOMAIN.into(),
            contract_revision: req.contract_revision,
            run_id: req.run_id,
            purpose: req.purpose,
            engine: req.engine,
            resources: req.resources,
            coordinator: req.coordinator,
            wall_time_max_ms: req.wall_time_max_ms,
            cleanup_max_ms: req.cleanup_max_ms,
            output_bytes_max: req.output_bytes_max,
            agent_renewable: req.agent_renewable,
            provenance: req.provenance,
            startup_document: ArtifactRef {
                path: req.outputs.startup.clone(),
                sha256: hex(&Sha256::digest(&startup)),
                bytes: startup.len() as u64,
                source: req.outputs.startup_source,
                license: req.outputs.startup_license,
            },
            startup_configuration_sha256,
            startup_search_configuration_sha256: hex(&typed
                .search_configuration_digest()
                .map_err(|e| format!("search identity: {e:?}"))?),
            own_r_root_tie: req.own_r_root_tie,
            system_runtime_lock: sys_lock.clone(),
        };
        let canonical_manifest = manifest
            .canonical_bytes()
            .map_err(|e| format!("manifest validation: {e:?}"))?;
        let mut manifest_value =
            serde_json::to_value(&manifest).map_err(|e| format!("manifest JSON: {e}"))?;
        manifest_value.sort_all_objects();
        let manifest_bytes =
            serde_json::to_vec(&manifest_value).map_err(|e| format!("manifest JSON: {e}"))?;
        let lock = manifest
            .lock()
            .map_err(|e| format!("lock validation: {e:?}"))?;
        typed
            .admit_against_v3(&lock)
            .map_err(|e| format!("typed startup admission: {e:?}"))?;
        doc.validate_against(&lock, &startup)
            .map_err(|e| format!("startup byte identity: {e:?}"))?;
        let runtime_bytes = sys_lock
            .canonical_bytes()
            .map_err(|e| format!("runtime canonical bytes: {e:?}"))?;
        let lock_bytes = serde_json::to_vec(&lock).map_err(|e| format!("lock JSON: {e}"))?;
        // All five files must be new. Remove only files opened by this process if
        // publication fails; no pre-existing input or evidence is touched.
        let files = [
            (req.outputs.startup, startup),
            (req.outputs.runtime_lock, runtime_bytes),
            (req.outputs.manifest, manifest_bytes),
            (req.outputs.manifest_canonical, canonical_manifest),
            (req.outputs.input_lock, lock_bytes),
        ];
        let mut created: Vec<(PathBuf, u64, u64)> = Vec::new();
        for (name, bytes) in files {
            let path = output_path(&root, &name)?;
            let result = (|| -> std::io::Result<()> {
                let mut file = OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&path)?;
                let metadata = file.metadata()?;
                created.push((path.clone(), metadata.dev(), metadata.ino()));
                file.write_all(&bytes)?;
                file.sync_all()
            })();
            if let Err(error) = result {
                for (path, device, inode) in created.iter().rev() {
                    if fs::symlink_metadata(path)
                        .is_ok_and(|m| m.is_file() && m.dev() == *device && m.ino() == *inode)
                    {
                        let _ = fs::remove_file(path);
                    }
                }
                return Err(format!(
                    "output publication failed: {error}; inspect output root for retained partial files"
                ));
            }
        }
        println!(
            "{{\"status\":\"registered\",\"system_identity\":\"{}\",\"runtime_lock_sha256\":\"{}\",\"execution_lock_sha256\":\"{}\",\"native_capture\":true,\"engine_model_asset_bytes_verified\":false,\"provider_ready\":false,\"gpu_inference\":false}}",
            hex(&captured.identity_sha256()),
            sys_lock.canonical_sha256,
            lock.canonical_sha256
        );
        Ok(())
    }
}

#[cfg(all(target_os = "linux", feature = "pals-adaptive", feature = "onnx-cuda"))]
fn main() {
    if let Err(e) = registration::run() {
        eprintln!("registration failed: {e}");
        std::process::exit(1);
    }
}
