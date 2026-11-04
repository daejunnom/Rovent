//! Explicit runtime-only diagnostic. No model, session, provider Run or NN.
//!
//! Native constructors can block or have unobserved GPU side effects. The
//! stage budget is checked before/after calls; an external tree supervisor
//! owns the hard timeout and must preserve stderr/EOF/exit evidence.
use rz_eval::asset;
use rz_eval::onnx::OrtRuntime;
use rz_eval::runtime_pin::{CudaRuntimeBundleSpec, RuntimeBundleFileRole, RuntimeCache};
use rz_native_loader::NativeLoadingProfile;
use serde::Deserialize;
use serde_json::{json, Value};
use std::error::Error;
use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::time::{Duration, Instant};

#[path = "support/native_startup_trace.rs"]
mod native_startup_trace;
use native_startup_trace::NativeStartupTrace;

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const FLAG: &str = "--runtime-loader-check=host-ort-v1";
const CONFIG_BYTES: usize = 64 * 1024;
const REPORT_BYTES: usize = 64 * 1024;

#[derive(Clone, Copy, Deserialize)]
enum LoadingProfile {
    #[serde(rename = "eager-cuda12-cudnn9-v1")]
    Eager,
    #[serde(rename = "experimental-cudnn-shim-lazy-v1")]
    Shim,
}
impl LoadingProfile {
    fn native(self) -> NativeLoadingProfile {
        match self {
            Self::Eager => NativeLoadingProfile::EagerCuda12Cudnn9V1,
            Self::Shim => NativeLoadingProfile::CuDnnShimLazyV1,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    runtime_library: PathBuf,
    runtime_library_sha256: String,
    cuda_bundle: PathBuf,
    cuda_bundle_sha256: String,
    runtime_cache: PathBuf,
    native_loading_profile: LoadingProfile,
    work_seconds: u64,
}

fn parse_args(args: impl IntoIterator<Item = OsString>) -> Result<(PathBuf, PathBuf)> {
    let values: Vec<_> = args.into_iter().collect();
    if values.len() != 3 || values[0] != FLAG {
        return Err("explicit --runtime-loader-check=host-ort-v1 CONFIG REPORT required".into());
    }
    let input = PathBuf::from(&values[1]);
    let output = PathBuf::from(&values[2]);
    plain_path(&input)?;
    plain_path(&output)?;
    if input == output {
        return Err("input and create-new report must differ".into());
    }
    Ok((input, output))
}

fn plain_path(path: &Path) -> Result<()> {
    if !path.is_absolute()
        || path
            .components()
            .any(|c| matches!(c, Component::ParentDir | Component::CurDir))
    {
        return Err("explicit absolute path without traversal required".into());
    }
    for component in path.components() {
        let name = component.as_os_str().to_string_lossy().to_ascii_lowercase();
        if name.starts_with(".env")
            || name.contains("credential")
            || name.contains("service-account")
            || name.contains("service_account")
            || matches!(
                name.as_str(),
                "id_rsa" | "id_ed25519" | "id_dsa" | "id_ecdsa"
            )
            || name.ends_with(".pem")
            || name.ends_with(".key")
        {
            return Err("secret-like diagnostic path refused".into());
        }
    }
    Ok(())
}

fn check_paths(config: &Config) -> Result<()> {
    for path in [
        &config.runtime_library,
        &config.cuda_bundle,
        &config.runtime_cache,
    ] {
        plain_path(path)?;
    }
    for digest in [&config.runtime_library_sha256, &config.cuda_bundle_sha256] {
        if digest.len() != 64
            || !digest
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err("explicit lowercase SHA-256 required".into());
        }
    }
    if !(1..=600).contains(&config.work_seconds) {
        return Err("finite stage budget from 1 through 600 seconds required".into());
    }
    Ok(())
}

fn new_report(path: &Path) -> Result<File> {
    let parent = path.parent().ok_or("report parent absent")?;
    for ancestor in parent.ancestors() {
        if fs::symlink_metadata(ancestor)?.file_type().is_symlink()
            || ancestor.join(".git").exists()
        {
            return Err("report requires an existing nonlink parent outside Git".into());
        }
    }
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    Ok(options.open(path)?)
}

fn budget(until: Instant) -> Result<()> {
    if Instant::now() >= until {
        return Err("original diagnostic stage budget exhausted; no next stage".into());
    }
    Ok(())
}

fn entered(
    trace: &NativeStartupTrace,
    stages: &mut Vec<&'static str>,
    stage: &'static str,
    until: Instant,
) -> Result<()> {
    budget(until)?;
    trace.stage(stage);
    stages.push(stage);
    Ok(())
}

fn returned(
    trace: &NativeStartupTrace,
    stages: &mut Vec<&'static str>,
    stage: &'static str,
    until: Instant,
) -> Result<()> {
    // Preserve the fact that the original call returned even if its budget
    // expired while it blocked. This is not an interruptible native deadline.
    trace.stage(stage);
    stages.push(stage);
    budget(until)
}

fn load(
    config: &Config,
    started: Instant,
    trace: &NativeStartupTrace,
    stages: &mut Vec<&'static str>,
) -> Result<Value> {
    if !cfg!(target_os = "linux") {
        return Err("runtime-only CUDA loader diagnostic requires Linux".into());
    }
    let until = started + Duration::from_secs(config.work_seconds);
    entered(trace, stages, "runtime_pin_entered", until)?;
    let cache = RuntimeCache::open(&config.runtime_cache)?;
    let bundle = asset::read_bounded(&config.cuda_bundle, CONFIG_BYTES)?;
    if hex(asset::sha256(&bundle)) != config.cuda_bundle_sha256 {
        return Err("actual CUDA bundle byte pin differs".into());
    }
    let spec = CudaRuntimeBundleSpec::from_json(std::str::from_utf8(&bundle)?)?;
    let core = spec
        .files
        .iter()
        .find(|f| f.role == RuntimeBundleFileRole::Core)
        .ok_or("CUDA core absent")?;
    if core.sha256 != config.runtime_library_sha256
        || config.runtime_library.file_name().and_then(|v| v.to_str())
            != Some(core.filename.as_str())
    {
        return Err("actual CUDA core filename/byte pin differs".into());
    }
    let pin = cache.cuda_bundle(
        config
            .runtime_library
            .parent()
            .ok_or("runtime parent absent")?,
        &spec,
    )?;
    returned(trace, stages, "runtime_pin_returned", until)?;
    entered(trace, stages, "runtime_load_entered", until)?;
    let runtime =
        OrtRuntime::load_with_cuda_loading_profile(&pin, config.native_loading_profile.native())?;
    returned(trace, stages, "runtime_load_returned", until)?;
    if hex(runtime.binary_digest()) != config.runtime_library_sha256 {
        return Err("loaded ORT binary pin differs".into());
    }
    entered(trace, stages, "runtime_origin_audit_entered", until)?;
    // CUDA provider images may load only at session/Run. Requiring their maps
    // here would make a model execution an accidental loader-only prerequisite.
    let observed = runtime.cuda_mapping_observation(false)?;
    returned(trace, stages, "runtime_origin_audit_returned", until)?;
    Ok(json!({
        "runtime_binary_sha256":hex(runtime.binary_digest()),
        "runtime_bundle_sha256":runtime.bundle_digest().map(hex),
        "ort_build_info":runtime.build_info(),
        "native_loading_profile":observed.profile.identifier(),
        "native_loading_profile_sha256":runtime.native_loading_profile_digest().map(hex),
        "declared_nvidia_files":observed.declared_nvidia_files,
        "required_nvidia_files":observed.required_nvidia_files,
        "mapped_nvidia_files":observed.mapped_nvidia_files,
        "deferred_nvidia_not_mapped":observed.deferred_nvidia_not_mapped,
        "mapped_ort_files":observed.mapped_ort_files,
        "mapping_scope":"NVIDIA dependencies before sessions; not provider/kernel placement"
    }))
}

fn hex(value: [u8; 32]) -> String {
    value.iter().map(|v| format!("{v:02x}")).collect()
}

fn bounded(value: String) -> String {
    let mut end = value.len().min(4096);
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    value[..end].to_owned()
}

fn run() -> Result<i32> {
    let started = Instant::now();
    let (config_path, report_path) = parse_args(std::env::args_os().skip(1))?;
    let mut report_file = new_report(&report_path)?;
    let trace = NativeStartupTrace::explicit();
    let mut stages = Vec::new();
    let mut requested = None;
    let result = (|| -> Result<Value> {
        let raw = asset::read_bounded(&config_path, CONFIG_BYTES)?;
        let config: Config = serde_json::from_slice(&raw)?;
        check_paths(&config)?;
        requested = Some(json!({
            "config_sha256":hex(asset::sha256(&raw)),
            "runtime_library":config.runtime_library,
            "runtime_library_sha256":config.runtime_library_sha256,
            "cuda_bundle":config.cuda_bundle,
            "cuda_bundle_sha256":config.cuda_bundle_sha256,
            "runtime_cache":config.runtime_cache,
            "native_loading_profile":config.native_loading_profile.native().identifier(),
            "work_seconds":config.work_seconds,
            "scope":"requested input declarations; actual load observation is separate"
        }));
        load(&config, started, &trace, &mut stages)
    })();
    let original_error = result.as_ref().err().map(|e| {
        eprintln!("runtime loader diagnostic failed: {e}");
        json!({"display":bounded(e.to_string()),"debug":bounded(format!("{e:?}"))})
    });
    let passed = result.is_ok();
    let report = json!({
        "schema":"rz-runtime-loader-diagnostic/1",
        "scope":"runtime_pin_and_native_environment_only",
        "status":if passed {"RuntimeInitializationReturned"} else {"FailedOrUnknown"},
        "requested":requested,
        "last_returned_or_entered_stages":stages,
        "original_error":original_error,
        "runtime_observation":result.ok(),
        "deadline_scope":"stage boundaries only; blocking native calls require external timeout",
        "requires_external_process_tree_supervisor":true,
        "external_process_exit_EOF_and_tree_gone":"not_observed_by_this_receipt",
        "model_loaded":false,"session_created":false,"NN_execution":false,
        "GPU_constructor_side_effects":"unknown",
        "GPU_provider_initialized":"not_claimed",
        "physical_completion":"unknown","kernel_placement_observed":false,
        "whole_product_acceptance":false,"numerical_acceptance":false,
        "performance_evidence":false,"training_executed":false
    });
    let raw = serde_json::to_vec(&report)?;
    if raw.len() >= REPORT_BYTES {
        return Err(
            "bounded diagnostic report exceeded its 64KiB cap; original stderr retained".into(),
        );
    }
    if let Err(error) = report_file
        .write_all(&raw)
        .and_then(|()| report_file.write_all(b"\n"))
        .and_then(|()| report_file.sync_all())
    {
        eprintln!("secondary runtime diagnostic report publication failure: {error}");
        return Err(error.into());
    }
    Ok(if passed { 0 } else { 2 })
}

fn main() {
    match run() {
        Ok(code) => std::process::exit(code),
        Err(error) => {
            eprintln!("runtime loader diagnostic admission/report failure: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn requires_explicit_flag_and_two_distinct_absolute_paths() {
        let args = |values: &[&str]| {
            values
                .iter()
                .map(|value| OsString::from(*value))
                .collect::<Vec<_>>()
        };
        let root = std::env::temp_dir();
        let input = root.join("runtime-loader-config.json");
        let output = root.join("runtime-loader-report.json");
        assert!(parse_args(vec![
            FLAG.into(),
            input.as_os_str().into(),
            output.as_os_str().into()
        ])
        .is_ok());
        assert!(parse_args(args(&[])).is_err());
        assert!(parse_args(args(&[FLAG, "relative.json", "report.json"])).is_err());
        assert!(parse_args(vec![
            FLAG.into(),
            input.as_os_str().into(),
            input.as_os_str().into()
        ])
        .is_err());
        assert!(parse_args(vec![
            "--other".into(),
            input.as_os_str().into(),
            output.as_os_str().into()
        ])
        .is_err());
    }
}
