//! One synchronous CUDA Run using caller-supplied system libraries and ONNX.
//!
//! Input JSON supplies `libraries` (role/path/required_mapping), finite capture
//! limits, a model path, profiling prefix, CUDA device and arena size. The model
//! is an independently supplied tiny FP32 Relu graph: X[1,4] -> Y[1,4]. This
//! diagnostic is not a benchmark or a model/engine acceptance. A supervisor
//! must bound native calls and preserve stdout/stderr and the generated profile.

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("system CUDA smoke requires Linux");
    std::process::exit(1);
}

#[cfg(target_os = "linux")]
mod linux {
    use ort::execution_providers::{CUDAExecutionProvider, ExecutionProvider};
    use ort::session::{builder::GraphOptimizationLevel, Session};
    use ort::tensor::TensorElementType;
    use ort::value::{Tensor, ValueType};
    use rz_eval::asset;
    use rz_eval::onnx::{
        verify_cuda_profile, OrtRuntime, SystemCaptureLimits, SystemLibraryRequest, SystemRuntime,
    };
    use serde::Deserialize;
    use serde_json::{json, Value};
    use std::error::Error;
    use std::path::{Path, PathBuf};
    use std::time::Duration;

    type Result<T> = std::result::Result<T, Box<dyn Error>>;

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Library {
        role: String,
        path: PathBuf,
        required_mapping: bool,
    }

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Config {
        model_path: PathBuf,
        profiling_prefix: PathBuf,
        device_id: i32,
        arena_bytes: usize,
        max_libraries: usize,
        max_file_bytes: u64,
        max_total_bytes: u64,
        max_capture_ms: u64,
        libraries: Vec<Library>,
    }

    enum ForwardFailure {
        /// A failed native Run does not prove that device work stopped.
        Unfenced(String),
        /// Run returned successfully; output extraction is a separate failure.
        Fenced(String),
    }

    fn forward(
        session: &mut Session,
        input: &Tensor<f32>,
    ) -> std::result::Result<Vec<f32>, ForwardFailure> {
        let outputs = session
            .run(ort::inputs!["X" => input])
            .map_err(|error| ForwardFailure::Unfenced(error.to_string()))?;
        let output = outputs
            .get("Y")
            .ok_or_else(|| ForwardFailure::Fenced("Y output missing".into()))?;
        let (shape, values) = output
            .try_extract_tensor::<f32>()
            .map_err(|error| ForwardFailure::Fenced(error.to_string()))?;
        if shape.as_ref() != [1, 4] {
            return Err(ForwardFailure::Fenced("Y shape is not [1,4]".into()));
        }
        Ok(values.to_vec())
    }

    fn one_tensor(value: &ValueType) -> bool {
        matches!(value, ValueType::Tensor { ty: TensorElementType::Float32, shape, .. } if shape.as_ref() == [1, 4])
    }

    fn hex(value: [u8; 32]) -> String {
        value.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    fn run(config: Config, report: &mut Value) -> Result<()> {
        if !config.model_path.is_absolute()
            || !config.profiling_prefix.is_absolute()
            || config.device_id < 0
            || config.arena_bytes == 0
            || config.max_capture_ms == 0
            || config
                .libraries
                .iter()
                .any(|library| !library.path.is_absolute())
        {
            return Err("absolute model/library/profile paths and positive limits required".into());
        }
        let requests = config
            .libraries
            .into_iter()
            .map(|library| SystemLibraryRequest {
                role: library.role,
                path: library.path,
                required_mapping: library.required_mapping,
            })
            .collect();
        report["stage"] = json!("capture_system_runtime");
        let owner = SystemRuntime::capture(
            requests,
            SystemCaptureLimits {
                max_libraries: config.max_libraries,
                max_file_bytes: config.max_file_bytes,
                max_total_bytes: config.max_total_bytes,
                max_elapsed: Duration::from_millis(config.max_capture_ms),
            },
        )?;
        report["system_runtime_identity_sha256"] = json!(hex(owner.identity_sha256()));
        report["stage"] = json!("load_ort");
        let runtime = OrtRuntime::load_system_cuda(&owner)?;
        report["ort_build_info"] = json!(runtime.build_info());
        if !runtime.is_system_cuda() || !runtime.is_cuda() {
            return Err("loaded runtime is not explicit system CUDA".into());
        }
        let cuda = CUDAExecutionProvider::default();
        if !cuda.is_available()? {
            return Err("CUDA EP unavailable; CPU fallback forbidden".into());
        }
        report["stage"] = json!("create_session");
        let model = asset::read_bounded(&config.model_path, 4 * 1024 * 1024)?;
        report["model_sha256"] = json!(hex(asset::sha256(&model)));
        let mut session = Session::builder()?
            .with_no_environment_execution_providers()?
            .with_intra_threads(1)?
            .with_inter_threads(1)?
            .with_parallel_execution(false)?
            .with_optimization_level(GraphOptimizationLevel::Disable)?
            .with_config_entry("session.disable_cpu_ep_fallback", "1")?
            .with_execution_providers([cuda
                .with_device_id(config.device_id)
                .with_memory_limit(config.arena_bytes)
                .with_tf32(false)
                .build()
                .error_on_failure()])?
            .with_profiling(&config.profiling_prefix)?
            .commit_from_memory(&model)?;
        report["session_created"] = json!(true);
        if session.inputs.len() != 1
            || session.outputs.len() != 1
            || session.inputs[0].name != "X"
            || session.outputs[0].name != "Y"
            || !one_tensor(&session.inputs[0].input_type)
            || !one_tensor(&session.outputs[0].output_type)
        {
            return Err("model must expose one FP32 X[1,4] input and Y[1,4] output".into());
        }
        runtime.verify_cuda_dependencies()?;
        let input = Tensor::from_array(([1usize, 4], vec![-2.0_f32, 0.0, 2.0, 3.0]))?;
        report["stage"] = json!("synchronous_run");
        let output = match forward(&mut session, &input) {
            Ok(values) => {
                report["physical_completion"] = json!("synchronous_run_returned");
                values
            }
            Err(ForwardFailure::Unfenced(error)) => {
                // Preserve the session, input and CUDA runtime until process exit.
                std::mem::forget(session);
                std::mem::forget(input);
                return Err(format!("ORT Run failed; physical completion unknown: {error}").into());
            }
            Err(ForwardFailure::Fenced(error)) => {
                report["physical_completion"] = json!("synchronous_run_returned");
                return Err(format!("Run returned but output extraction failed: {error}").into());
            }
        };
        report["forward_output"] = json!(&output);
        let expected = [0.0_f32, 0.0, 2.0, 3.0];
        let numerical_match = output.len() == expected.len()
            && output
                .iter()
                .zip(expected)
                .all(|(a, b)| a.is_finite() && (*a - b).abs() <= 1e-6);
        report["forward_expected_match"] = json!(numerical_match);
        report["stage"] = json!("mapping_and_cuda_placement");
        runtime.verify_cuda_runtime_mappings()?;
        let mapping = owner.verify_mappings()?;
        report["mapped_roles"] = json!(mapping.mapped_roles);
        let profile = session.end_profiling()?;
        let profile_path = Path::new(&profile);
        let wanted_parent = config
            .profiling_prefix
            .parent()
            .ok_or("profile parent absent")?
            .canonicalize()?;
        let metadata = std::fs::symlink_metadata(profile_path)?;
        if !profile_path.is_absolute()
            || !metadata.is_file()
            || metadata.file_type().is_symlink()
            || profile_path
                .parent()
                .ok_or("profile path parent absent")?
                .canonicalize()?
                != wanted_parent
        {
            return Err("generated profile is outside the requested directory".into());
        }
        let profile_bytes = asset::read_bounded(profile_path, 4 * 1024 * 1024)?;
        let cuda_nodes = verify_cuda_profile(&profile_bytes)?;
        report["profile_path"] = json!(profile);
        report["executed_cuda_nodes"] = json!(cuda_nodes);
        report["profile_sha256"] = json!(hex(asset::sha256(&profile_bytes)));
        if !numerical_match {
            return Err("Relu output differs from the fixed expected result".into());
        }
        report["stage"] = json!("complete");
        Ok(())
    }

    pub fn main() {
        let mut report = json!({"schema":"rz-system-cuda-smoke/1", "status":"failed_or_unknown",
            "stage":"configuration", "session_created":false, "physical_completion":"unknown",
            "forward_expected_match":null, "performance_evidence":false, "training_executed":false});
        let result = (|| -> Result<()> {
            let mut args = std::env::args_os().skip(1);
            let path = PathBuf::from(args.next().ok_or("expected absolute JSON config path")?);
            if args.next().is_some() || !path.is_absolute() {
                return Err("expected exactly one absolute JSON config path".into());
            }
            let bytes = asset::read_bounded(&path, 64 * 1024)?;
            let config: Config = serde_json::from_slice(&bytes)?;
            run(config, &mut report)
        })();
        match result {
            Ok(()) => report["status"] = json!("passed"),
            Err(error) => report["error"] = json!(error.to_string()),
        }
        println!("{}", report);
        if report["status"] != "passed" {
            std::process::exit(1);
        }
    }
}

#[cfg(target_os = "linux")]
fn main() {
    linux::main();
}
