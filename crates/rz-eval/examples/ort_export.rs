//! Separate, opt-in conversion with the same pinned CUDA/FP32/Level1 path.
//! Neither a throughput benchmark nor independent numerical acceptance.
#[cfg(target_os = "linux")]
fn admit_legacy_cuda_bundle() -> Result<(), Box<dyn std::error::Error>> {
    Err("legacy CUDA bundle execution is disabled; use a system CUDA runtime".into())
}

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use rz_eval::{
        asset::{self, MaiaAsset},
        onnx::{BackendConfig, OnnxBackend, OrtRuntime, Provider},
        ort_model::OrtModelManifest,
        runtime_pin::{CudaRuntimeBundleSpec, RuntimeCache},
    };
    use std::{fs::OpenOptions, io::Write, path::Path};
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 7 || args.iter().any(|a| !Path::new(a).is_absolute()) {
        return Err("usage: ort_export SOURCE ONNX EXPORT CUDA_CORE CUDA_BUNDLE NEW_ORT NEW_MANIFEST (absolute; outputs outside Git)".into());
    }
    // Reject the retired profile before reserving either output file.
    admit_legacy_cuda_bundle()?;
    let root = Path::new(&args[5])
        .parent()
        .ok_or("output parent missing")?
        .canonicalize()?;
    if root.ancestors().any(|p| p.join(".git").exists())
        || Path::new(&args[6])
            .parent()
            .ok_or("manifest parent missing")?
            .canonicalize()?
            != root
    {
        return Err("fresh output files must share a caller-owned parent outside Git".into());
    }
    let output = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&args[5])?;
    let mut manifest_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[6])?;
    let spec = CudaRuntimeBundleSpec::from_json(std::str::from_utf8(&asset::read_bounded(
        Path::new(&args[4]),
        64 * 1024,
    )?)?)?;
    let core = spec
        .files
        .iter()
        .find(|f| f.role == rz_eval::runtime_pin::RuntimeBundleFileRole::Core)
        .ok_or("bundle core missing")?;
    if Path::new(&args[3]).file_name().and_then(|v| v.to_str()) != Some(core.filename.as_str()) {
        return Err("declared CUDA core differs".into());
    }
    let cache = RuntimeCache::for_user()?;
    let pin = cache.cuda_bundle(
        Path::new(&args[3]).parent().ok_or("core parent missing")?,
        &spec,
    )?;
    let runtime = OrtRuntime::load(&pin)?;
    let asset = MaiaAsset::load(
        Path::new(&args[0]),
        Path::new(&args[1]),
        Path::new(&args[2]),
    )?;
    let mut config = BackendConfig::cpu();
    config.provider = Provider::Cuda {
        device_id: 0,
        arena_bytes: asset.profile().cuda_arena_bytes(),
    };
    let placement = root.join("export-placement");
    std::fs::create_dir(&placement)?;
    config.profiling_prefix = Some(placement.join("placement"));
    let (source, backend) = OnnxBackend::export_owned_ort(&runtime, asset, config, &output)?;
    backend.verify_cuda_runtime_mappings()?;
    if backend.has_unconfirmed_physical_completion() {
        return Err("export probe completion unconfirmed".into());
    }
    output.sync_all()?;
    // This artifact is created once, outside the later A/B measurement window.
    let proc_path = format!("/proc/self/fd/{}", std::os::fd::AsRawFd::as_raw_fd(&output));
    let bytes = asset::read_bounded(Path::new(&proc_path), source.profile().max_onnx_bytes())?;
    if bytes.get(4..8) != Some(b"ORTM".as_slice()) {
        return Err("export is not ORT format".into());
    }
    let manifest = OrtModelManifest {
        schema: 1,
        source_onnx_sha256: source.manifest().onnx_sha256.clone(),
        source_export_manifest_sha256: hex(source.manifest_digest()),
        ort_sha256: hex(asset::sha256(&bytes)),
        ort_bytes: bytes.len(),
        runtime_core_sha256: hex(runtime.binary_digest()),
        runtime_bundle_sha256: hex(runtime.bundle_digest().ok_or("no pinned CUDA bundle")?),
        runtime_version: "1.22.0".into(),
        optimization_level: 1,
        provider: "cuda".into(),
        precision: "fp32".into(),
        tf32: false,
        redistribution_ready: false,
    };
    manifest.validate(&source)?;
    manifest_file.write_all(&serde_json::to_vec_pretty(&manifest)?)?;
    manifest_file.write_all(b"\n")?;
    manifest_file.sync_all()?;
    println!(
        "{}",
        serde_json::json!({"kind":"derived_ort_conversion","accepted":true,"independent_numerical_acceptance":false,
        "ort_bytes":manifest.ort_bytes,"ort_sha256":manifest.ort_sha256,"physical_completion_confirmed":true})
    );
    drop((backend, output));
    Ok(())
}
#[cfg(target_os = "linux")]
fn hex(bytes: [u8; 32]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
#[cfg(not(target_os = "linux"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err("pinned CUDA ORT export requires Linux; no assets or provider loaded".into())
}
