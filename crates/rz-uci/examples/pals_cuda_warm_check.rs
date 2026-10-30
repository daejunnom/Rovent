//! 유한한 실제 CUDA Warm 인수 실행기. 제품 검색/인코딩/추론은 공개 API를 호출한다.
//!
//! capture --export-manifest PATH --export-manifest-sha256 HEX64
//!   --cuda-bundle PATH --cuda-bundle-sha256 HEX64 --runtime-root DIRECTORY
//!   --runtime-output-root DIRECTORY --scenario PATH --scenario-sha256 HEX64
//!   --pals-cuda-warm-max-leases 1 --pals-cuda-warm-device-bytes-max BYTES
//!   --output-json NEW_PATH
//! 선택한 metadata-control만 허용할 때 다음 세 인자를 함께 제공한다:
//!   --control-inventory PATH --control-inventory-sha256 HEX64 --profile-root DIRECTORY
//!
//! verify-reference --capture-json PATH --capture-sha256 HEX64
//!   --independent-reference PATH --independent-reference-sha256 HEX64
//!   --output-json NEW_PATH
//!
//! 모든 경로는 절대 경로이며 checkout 밖이다. 실행은 Linux, GPU 0 한 대,
//! native physical worker 한 개, startup 두 forward를 포함한 사전등록 16..128
//! role forward(기본 64),
//! 작업 60초 + 정리 30초, 단일 JSON 산출물 128 MiB로 제한한다. 부모 감독자는
//! 프로세스/파이프에도 90초/128 MiB 상한을 적용해야 한다. 이 파일은 GPU 실행을
//! 예약하거나 export/reference를 생성하지 않는다. --runtime-output-root는 기존
//! caller-owned RuntimeCache의 root이며 capture 산출물 디렉터리와 분리한다.
//! owner 한도는 Native/Arena와 같은 명시 K/V payload bytes 단위다.
//! 등록 FLI 최대 S194의 FP32 K/V 두 쌍 중첩은 397312B이며 weights,
//! ORT workspace/staging 또는 전체 장치 VRAM을 포함하지 않는다.
//! capture의 실행 검사와 별도의
//! CPU Torch/ORT seeded reference 검사를 분리하고 최종 passed는 둘 다 요구한다.
//! scenario.execution_profile은 actual_opponent_continuation_v1(기본) 또는
//! iterative_frozen_model_wdl_v2다. 같은 실제 search 안에서 중단 CPU stack,
//! 수락된 GPU Repair/C tail, live token의 실제 scheduler resume가 연결되지
//! 않으면 두 profile 모두 인수 실패다. pause를 만들려고 NN/CPU 출력을 바꾸지 않는다.
//!
//! capture에 --cost-same-seed-context를 명시하면 독립 API 비용을 관측한다.
//! 실제 수락 Repair source를 보존하고 Warm/Fresh owner를 순차 실행하며, 각
//! lane의 같은 Repair 입력으로 준비 3회와 측정 5회를 45초 work 한도 안에 수행한다.
//! 비용 출력은 known physical completion 뒤 close_unconsumed로 닫는다. consumer는
//! 실제 0이며 기존 integration capture의 수락 증거와 구분한다. sample별 실제
//! public-cache/NN 계수가 미관측이면 not_comparable로 남기고 비율을 만들지 않는다.
//! 같은 seed의 독립 Torch/ORT reference 요구는 유지한다.

#![recursion_limit = "256"]

#[cfg(not(all(feature = "onnx-cuda", feature = "experimental-io-binding")))]
fn main() -> std::process::ExitCode {
    eprintln!("pals_cuda_warm_check requires onnx-cuda and experimental-io-binding");
    std::process::ExitCode::FAILURE
}

#[cfg(all(feature = "onnx-cuda", feature = "experimental-io-binding"))]
fn main() -> std::process::ExitCode {
    check::main()
}

#[cfg(all(feature = "onnx-cuda", feature = "experimental-io-binding"))]
mod check {
    use rz_contracts::RequestId;
    use rz_eval::asset;
    use rz_eval::error::BackendError;
    use rz_eval::onnx::{OrtRuntime, Provider};
    use rz_eval::pals_model::{PalsModelConfig, PalsModelInput, PalsModelProfile, PalsRawOutput};
    use rz_eval::pals_onnx::{
        PRIVATE_CUDA_WARM_MEASUREMENT_CONTRACT, PRIVATE_CUDA_WARM_SCHEMA, PalsCuDnnConvSearch,
        PalsCudaControlPolicy, PalsNativeResult, PalsOnnxConfig,
    };
    use rz_eval::pals_private::{PrivateInvocation, PrivateSeedProvenance};
    use rz_eval::runtime_pin::{CudaRuntimeBundleSpec, RuntimeCache};
    use rz_position::{BoardMove, Position, PositionIdentity, PositionSnapshot};
    use rz_search::cpu::{
        CpuCapabilities, CpuConfig, CpuEngine, CpuError, CpuLimits, CpuOrderingPolicy,
        CpuPausedStackSnapshot, CpuProfile, CpuReport, CpuResumePolicy, CpuResumeToken,
        CpuSearcher, CpuWork,
    };
    use rz_search::cpu_checker::{CheckerError, OwnedCpuChecker};
    use rz_search::cpu_value::{BootstrapCpuValue, CpuValueIdentity};
    use rz_search::pals::engine::{
        CpuInterleavePolicy, DivergenceQuery, ModelValueIdentity, ModelValueOutput, PalsConfig,
        PalsEngine, PalsError, PalsLimits, PostRepairRecheckPolicy, RecheckFinished,
        RecheckPrepared, ResolverPolicy, RoleAcceptance, RoleError, RoleEvaluation,
        RoleLogicalContext, RoleModel, RoleQuery, RoleQueryPurpose, RoleRecord, RoleSearchClosure,
    };
    use rz_search::pals::store::{ExecutionId, PalsStores, TaskQuestion};
    use rz_uci::pals_native::{
        NativeCudaWarmLoadError, NativeCudaWarmObservationLimits, NativeExecutionReceipt,
        NativeOwnerOptions, NativePreparedContext, NativePrivateWarmPrepared, NativeQueryKind,
        NativeRoleExecutionBinding, NativeRoleFinishHandle, NativeRoleModel, NativeRoleObserver,
        NativeRoleReceipt, NativeRoleRejection, NativeRoleTerminal, prepare_role_input_for_config,
    };
    use serde::Deserialize;
    use serde_json::{Value, json};
    use sha2::{Digest as _, Sha256};
    use std::collections::BTreeMap;
    use std::fs::OpenOptions;
    use std::io::{self, Write};
    use std::path::{Path, PathBuf};
    use std::process::ExitCode;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::{Arc, Mutex, mpsc};
    use std::time::{Duration, Instant};

    const SCHEMA: &str = "rz-pals-native-cuda-warm-check/1";
    const REFERENCE_SCHEMA: &str = "rz-pals-cuda-warm-actual-reference/1";
    const SCENARIO_SCHEMA: &str = "rz-pals-native-cuda-warm-scenario/1";
    const WALL: Duration = Duration::from_secs(60);
    const CLEANUP: Duration = Duration::from_secs(30);
    const OUTPUT_LIMIT: usize = 128 * 1024 * 1024;
    const MIN_ROLE_FORWARDS: u64 = 16;
    const MAX_ROLE_FORWARDS: u64 = 128;
    const DEFAULT_ROLE_FORWARDS: u64 = 64;
    const STARTUP_ROLE_FORWARDS: u64 = 2;
    const FOLLOWUP_ROLE_FORWARDS: u64 = 2;
    const CALL_OUTPUT_RESERVE: usize = 512 * 1024;
    const REPORT_HEADROOM: usize = 8 * 1024 * 1024;
    const MAX_RECHECK_EVENTS: usize = 64;
    const LATENT_ELEMENTS: usize = 6144;
    const ATOL: f64 = 1e-4;
    const RTOL: f64 = 1e-3;
    const POLICY_WDL_ATOL: f64 = 1e-4;
    // The latent and auxiliary heads use the same strict raw threshold. This is
    // an explicit declaration; historical Warm thresholds do not grant a pass.
    const LATENT_ATOL: f64 = 1e-4;
    const LATENT_RTOL: f64 = 1e-3;
    const MAX_CPU_TRACE_ATTEMPTS: usize = 128;
    const COST_PREPARATION_CALLS: usize = 3;
    const COST_MEASURED_CALLS: usize = 5;
    const COST_LANE_CALLS: usize = COST_PREPARATION_CALLS + COST_MEASURED_CALLS;
    const COST_EXTRA_ROLE_FORWARDS: u64 = COST_LANE_CALLS as u64 * 2 + STARTUP_ROLE_FORWARDS;
    const COST_WALL: Duration = Duration::from_secs(45);
    // DEFAULT cost includes two sequential owners' loading, readiness, samples
    // and physical joins. This opt-in bound changes no NN or sample budget.
    const COST_CONV_DEFAULT_WALL: Duration = Duration::from_secs(150);
    const COST_DEVICE_BYTES_MAX: u64 = 397_312;

    #[derive(Clone)]
    struct Failure {
        stage: &'static str,
        code: &'static str,
        load_cause: Option<Box<NativeCudaWarmLoadError>>,
        pals_cause: Option<Box<PalsError>>,
    }
    impl Failure {
        fn wire(self) -> Value {
            let mut wire = json!({"stage": self.stage, "code": self.code});
            if let Some(failure) = self.load_cause {
                // Static backend detail and bounded cause identity only. Raw
                // native messages, tensor buffers and private paths stay out.
                let backend = failure.backend.as_ref().map(|error| json!({
                    "kind":format!("{:?}",error.kind),"stage":format!("{:?}",error.stage),
                    "detail":error.detail,"cause":error.cause.map(|cause| json!({
                        "code":format!("{:?}",cause.code),"prefix_sha256":hex(&cause.prefix_sha256),
                        "hashed_bytes":cause.hashed_bytes,"truncated":cause.truncated,
                        "formatting_failed":cause.formatting_failed
                    })),"raw_native_text_published":false
                }));
                wire["load_diagnostic"] = json!({"stage":failure.stage,
                    "role_kind":role_error_code(&failure.cause),"backend":backend,
                    "original_cause_retained_until_projection":true,
                    "physical_run_or_completion_claim":false});
            }
            if let Some(error) = self.pals_cause {
                wire["pals_diagnostic"] = pals_error_diagnostic(&error);
            }
            wire
        }
    }
    fn fail(stage: &'static str, code: &'static str) -> Failure {
        Failure {
            stage,
            code,
            load_cause: None,
            pals_cause: None,
        }
    }
    fn require(condition: bool, stage: &'static str, code: &'static str) -> Result<(), Failure> {
        if condition {
            Ok(())
        } else {
            Err(fail(stage, code))
        }
    }
    fn role_failure(stage: &'static str, error: RoleError) -> Failure {
        fail(stage, role_error_code(&error))
    }
    fn role_error_code(error: &RoleError) -> &'static str {
        match error {
            RoleError::Unavailable => "unavailable",
            RoleError::Unsupported(_) => "unsupported",
            RoleError::InvalidOutput => "invalid_output",
            RoleError::Canceled => "canceled",
            RoleError::Deadline => "deadline",
            RoleError::PhysicalCompletionUnknown => "physical_completion_unknown",
            RoleError::Backend(_) => "backend_error",
        }
    }
    fn cuda_warm_load_failure(error: NativeCudaWarmLoadError) -> Failure {
        Failure {
            stage: "load_cuda_private_warm",
            code: role_error_code(&error.cause),
            load_cause: Some(Box::new(error)),
            pals_cause: None,
        }
    }
    fn pals_failure(stage: &'static str, code: &'static str, error: PalsError) -> Failure {
        Failure {
            stage,
            code,
            load_cause: None,
            pals_cause: Some(Box::new(error)),
        }
    }
    fn pals_error_diagnostic(error: &PalsError) -> Value {
        let kind = match error {
            PalsError::InvalidConfig => "InvalidConfig",
            PalsError::InvalidLimits => "InvalidLimits",
            PalsError::Rules(_) => "Rules",
            PalsError::Cpu(_) => "Cpu",
            PalsError::Checker(_) => "Checker",
            PalsError::Role(_) => "Role",
            PalsError::Store(_) => "Store",
            PalsError::Review(_) => "Review",
            PalsError::Rovent(_) => "Rovent",
            PalsError::Capacity => "Capacity",
            PalsError::RoleCallLimit => "RoleCallLimit",
        };
        // Only typed categories and compile-time checker detail are exposed.
        // In particular, do not format backend/CPU/Rules error payloads, which
        // can carry private paths or native messages. Keep the typed cause until
        // this projection instead of replacing a search failure with cleanup.
        let checker = match error {
            PalsError::Checker(CheckerError::Invalid(detail)) => {
                Some(json!({"kind":"Invalid","detail":detail}))
            }
            PalsError::Checker(CheckerError::Unsupported(detail)) => {
                Some(json!({"kind":"Unsupported","detail":detail}))
            }
            PalsError::Checker(CheckerError::OwnedReportRejected { reason }) => {
                Some(json!({"kind":"OwnedReportRejected","detail":reason}))
            }
            PalsError::Checker(CheckerError::External { stage, code }) => {
                Some(json!({"kind":"External","stage":stage,"code":code}))
            }
            PalsError::Checker(CheckerError::Owned(_)) => Some(json!({"kind":"Owned"})),
            _ => None,
        };
        let role = match error {
            PalsError::Role(error) => Some(json!({"kind":role_error_code(error),
                "raw_error_text_published":false})),
            _ => None,
        };
        json!({"kind":kind,"checker":checker,"role":role,
            "original_cause_retained_until_projection":true,"raw_error_text_published":false})
    }
    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }
    fn bits_digest(bits: &[u32]) -> String {
        let mut hash = Sha256::new();
        for value in bits {
            hash.update(value.to_le_bytes());
        }
        hex(&hash.finalize())
    }
    fn seed_provenance_digest(bits: &[u32]) -> String {
        // Match the bank's provenance framing; raw capture digests stay unframed.
        let mut hash = Sha256::new();
        hash.update(b"rz-pals-private-finite-fp32-latent/1");
        hash.update((bits.len() as u64).to_le_bytes());
        for value in bits {
            hash.update(value.to_le_bytes());
        }
        hex(&hash.finalize())
    }
    fn sha_json<T: serde::Serialize>(value: &T) -> Result<String, Failure> {
        serde_json::to_vec(value)
            .map(|bytes| hex(&Sha256::digest(bytes)))
            .map_err(|_| fail("capture", "serialization_failed"))
    }
    fn valid_digest(value: &str) -> bool {
        value.len() == 64
            && value
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    }
    fn request_id(id: RequestId) -> Value {
        json!({"epoch": id.epoch.0, "sequence": id.sequence})
    }
    fn moves(line: &[BoardMove]) -> Vec<String> {
        line.iter().map(ToString::to_string).collect()
    }
    fn input_f32_bits(input: &PalsModelInput) -> Value {
        json!({
            "metadata": input.metadata.map(f32::to_bits), "query": input.query.map(f32::to_bits),
            "records": input.records.iter().map(|record| record.features.map(f32::to_bits)).collect::<Vec<_>>(),
            "divergence_features": input.divergence_features.iter().map(|features| features.map(f32::to_bits)).collect::<Vec<_>>(),
        })
    }
    fn numeric_contract() -> Value {
        json!({"raw_logits": {"atol": ATOL, "rtol": RTOL},
            "legal_policy_and_wdl": {"max_abs": POLICY_WDL_ATOL},
            "private_latent": {"atol": LATENT_ATOL, "rtol": LATENT_RTOL},
            "divergence_and_task_heads": {"atol": ATOL, "rtol": RTOL},
            "historical_looser_warm_thresholds_used": false})
    }
    fn logical(context: &RoleLogicalContext) -> Value {
        json!({
            "game_generation": context.game_generation, "search_generation": context.search_generation,
            "situation": context.situation, "state": context.state, "focus": context.focus,
            "purpose": format!("{:?}", context.purpose), "prefix": moves(&context.prefix),
            "focus_sha256_hex": hex(&context.focus_sha256), "prefix_sha256_hex": hex(&context.prefix_sha256),
            "proposal_sha256_hex": hex(&context.proposal_sha256),
            "refutation_sha256_hex": context.refutation_sha256.map(|v| hex(&v)),
            "divergence_sha256_hex": hex(&context.divergence_sha256),
            "public_revision": context.public_revision, "situation_revision": context.situation_revision
        })
    }
    fn invocation(value: PrivateInvocation) -> Value {
        match value {
            PrivateInvocation::Fresh { input_key } => json!({
                "mode": "fresh", "input_key_hex": hex(&input_key), "invocation_key_hex": hex(&input_key)
            }),
            PrivateInvocation::ApproxCudaWarmV2 {
                input_key,
                seed_seal,
                invocation_key,
            } => json!({
                "mode": "approx_cuda_warm_v2", "input_key_hex": hex(&input_key),
                "seed_seal_hex": hex(&seed_seal), "invocation_key_hex": hex(&invocation_key)
            }),
            PrivateInvocation::ApproxWarmV1 {
                input_key,
                seed_seal,
                invocation_key,
            } => json!({
                "mode": "approx_warm_v1_refused", "input_key_hex": hex(&input_key),
                "seed_seal_hex": hex(&seed_seal), "invocation_key_hex": hex(&invocation_key)
            }),
        }
    }
    fn provenance(value: &PrivateSeedProvenance) -> Value {
        json!({
            "role": value.role, "model_manifest_sha256_hex": hex(&value.model.model),
            "encoding_semantic_sha256_hex": hex(&value.model.encoding),
            "model_epoch_hex": hex(&value.model.model_epoch), "frozen_epoch": value.model.frozen_epoch,
            "game_generation": value.game_generation, "search_generation": value.search_generation,
            "source_input_hex": hex(&value.source_input), "source_history_hex": hex(&value.source_history),
            "source_non_record_context_hex": hex(&value.source_non_record_context),
            "source_record_revision": value.source_record_revision, "source_rules_revision": value.source_rules_revision,
            "source_situation_debug": format!("{:?}", value.source_situation),
            "source_invocation": invocation(value.source_invocation), "source_lease_id": value.source_lease_id,
            "seed_sequence": value.seed_sequence, "latent_bits_digest_hex": hex(&value.latent_bits_digest),
            "seal_hex": hex(&value.seal)
        })
    }

    struct Args {
        mode: String,
        values: BTreeMap<String, String>,
        output: PathBuf,
        cost_same_seed_context: bool,
        cost_conv_default: bool,
    }
    impl Args {
        fn value(&self, key: &str) -> Result<&str, Failure> {
            self.values
                .get(key)
                .map(String::as_str)
                .ok_or(fail("arguments", "required_input_missing"))
        }
        fn path(&self, key: &str) -> Result<PathBuf, Failure> {
            Ok(self.value(key)?.into())
        }
        fn cuda_warm_limits(&self) -> Result<NativeCudaWarmObservationLimits, Failure> {
            cuda_warm_limits(
                self.value("--pals-cuda-warm-max-leases")?,
                self.value("--pals-cuda-warm-device-bytes-max")?,
            )
        }
    }
    fn cuda_warm_limits(
        max_leases: &str,
        device_bytes_max: &str,
    ) -> Result<NativeCudaWarmObservationLimits, Failure> {
        require(
            max_leases == "1"
                && !device_bytes_max.is_empty()
                && device_bytes_max.len() <= 20
                && device_bytes_max.bytes().all(|byte| byte.is_ascii_digit()),
            "arguments",
            "cuda_warm_limits_invalid",
        )?;
        let device_bytes_max = device_bytes_max
            .parse::<u64>()
            .map_err(|_| fail("arguments", "cuda_warm_limits_invalid"))?;
        require(
            device_bytes_max > 0,
            "arguments",
            "cuda_warm_limits_invalid",
        )?;
        Ok(NativeCudaWarmObservationLimits {
            max_leases: 1,
            device_bytes_max,
        })
    }
    fn external(path: &Path, new_file: bool) -> Result<(), Failure> {
        require(path.is_absolute(), "arguments", "absolute_path_required")?;
        let forbidden = path
            .components()
            .filter_map(|part| part.as_os_str().to_str())
            .any(|part| {
                let name = part.to_ascii_lowercase();
                name == ".env"
                    || name.starts_with(".env.")
                    || name.contains("credential")
                    || name.contains("service-account")
                    || name.contains("api_key")
                    || name.starts_with("id_ed25519")
                    || name.starts_with("id_rsa")
            });
        require(!forbidden, "arguments", "secret_path_refused")?;
        let checkout = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .and_then(|path| path.canonicalize().ok())
            .ok_or(fail("arguments", "checkout_boundary_unavailable"))?;
        let resolved = if new_file {
            require(
                matches!(std::fs::symlink_metadata(path), Err(error) if error.kind() == std::io::ErrorKind::NotFound),
                "arguments",
                "output_exists",
            )?;
            path.parent().and_then(|path| path.canonicalize().ok())
        } else {
            path.canonicalize().ok()
        }
        .ok_or(fail("arguments", "input_or_parent_unavailable"))?;
        require(
            !resolved.starts_with(checkout),
            "arguments",
            "checkout_path_refused",
        )
    }
    fn args() -> Result<Args, Failure> {
        let mut arguments = std::env::args().skip(1);
        let mode = arguments.next().ok_or(fail("arguments", "mode_required"))?;
        let allowed: &[&str] = match mode.as_str() {
            "capture" => &[
                "--export-manifest",
                "--export-manifest-sha256",
                "--cuda-bundle",
                "--cuda-bundle-sha256",
                "--runtime-root",
                "--runtime-output-root",
                "--scenario",
                "--scenario-sha256",
                "--pals-cuda-warm-max-leases",
                "--pals-cuda-warm-device-bytes-max",
                "--output-json",
                "--control-inventory",
                "--control-inventory-sha256",
                "--profile-root",
                "--cudnn-conv-search",
            ],
            "verify-reference" => &[
                "--capture-json",
                "--capture-sha256",
                "--independent-reference",
                "--independent-reference-sha256",
                "--output-json",
            ],
            _ => return Err(fail("arguments", "unsupported_mode")),
        };
        let mut values = BTreeMap::new();
        let mut cost_same_seed_context = false;
        let mut cost_conv_default = false;
        while let Some(flag) = arguments.next() {
            if matches!(
                flag.as_str(),
                "--cost-same-seed-context" | "--cost-same-seed-context-conv-default"
            ) {
                cost_opt_in(&mode, &mut cost_same_seed_context)?;
                cost_conv_default = flag == "--cost-same-seed-context-conv-default";
                continue;
            }
            require(
                values.len() < allowed.len() && allowed.contains(&flag.as_str()),
                "arguments",
                "unknown_option",
            )?;
            let value = arguments.next().ok_or(fail("arguments", "missing_value"))?;
            require(
                !value.is_empty() && value.len() <= 4096 && !values.contains_key(&flag),
                "arguments",
                "invalid_or_duplicate_option",
            )?;
            if flag.ends_with("-sha256") {
                require(
                    valid_digest(&value),
                    "arguments",
                    "digest_must_be_lowercase_hex64",
                )?;
            }
            values.insert(flag, value);
        }
        let output = values
            .get("--output-json")
            .ok_or(fail("arguments", "output_required"))?
            .into();
        let args = Args {
            mode,
            values,
            output,
            cost_same_seed_context,
            cost_conv_default,
        };
        for &flag in allowed {
            let optional = [
                "--control-inventory",
                "--control-inventory-sha256",
                "--profile-root",
                "--cudnn-conv-search",
            ]
            .contains(&flag);
            if !optional {
                args.value(flag)?;
            }
            if !flag.ends_with("-sha256")
                && !matches!(
                    flag,
                    "--pals-cuda-warm-max-leases"
                        | "--pals-cuda-warm-device-bytes-max"
                        | "--cudnn-conv-search"
                )
            {
                if let Some(value) = args.values.get(flag) {
                    external(
                        Path::new(value),
                        matches!(flag, "--output-json" | "--profile-root"),
                    )?;
                }
            }
        }
        let control_fields = [
            "--control-inventory",
            "--control-inventory-sha256",
            "--profile-root",
        ]
        .iter()
        .filter(|flag| args.values.contains_key(**flag))
        .count();
        require(
            control_fields == 0 || control_fields == 3,
            "arguments",
            "control_inventory_requires_all_three_fields",
        )?;
        cuda_conv_selection(&args)?;
        cost_default_environment(&args, std::env::var("NVIDIA_TF32_OVERRIDE").ok().as_deref())?;
        if args.mode == "capture" {
            let limits = args.cuda_warm_limits()?;
            if args.cost_same_seed_context {
                require(
                    limits.max_leases == 1 && limits.device_bytes_max == COST_DEVICE_BYTES_MAX,
                    "arguments",
                    "cost_requires_registered_exclusive_cuda_warm_limits",
                )?;
            }
        }
        Ok(args)
    }
    fn cost_opt_in(mode: &str, enabled: &mut bool) -> Result<(), Failure> {
        require(
            mode == "capture" && !*enabled,
            "arguments",
            "cost_same_seed_context_invalid_or_duplicate",
        )?;
        *enabled = true;
        Ok(())
    }
    fn cuda_conv_selection(args: &Args) -> Result<PalsCuDnnConvSearch, Failure> {
        match args.values.get("--cudnn-conv-search").map(String::as_str) {
            None if !args.cost_conv_default => Ok(PalsCuDnnConvSearch::LegacyImplicitExhaustive),
            Some("default-device-warm-v1")
                if args.mode == "capture"
                    && (!args.cost_same_seed_context || args.cost_conv_default)
                    && [
                        "--control-inventory",
                        "--control-inventory-sha256",
                        "--profile-root",
                    ]
                    .iter()
                    .all(|key| args.values.contains_key(*key)) =>
            {
                let config = PalsOnnxConfig {
                    provider: Provider::Cuda {
                        device_id: 0,
                        arena_bytes: 256 * 1024 * 1024,
                    },
                    device_public_memory: true,
                    cudnn_conv_search: PalsCuDnnConvSearch::ExperimentalDefaultDeviceWarm,
                    ..PalsOnnxConfig::cpu()
                };
                config
                    .validate_cuda_warm_execution_selection(
                        Some(rz_eval::onnx::NativeLoadingProfile::EagerCuda12Cudnn9V1),
                        true,
                    )
                    .map_err(|_| fail("arguments", "invalid_default_device_warm_admission"))?;
                Ok(PalsCuDnnConvSearch::ExperimentalDefaultDeviceWarm)
            }
            _ => Err(fail(
                "arguments",
                "default_device_warm_requires_separate_eager_control_capture",
            )),
        }
    }
    fn cost_default_environment(args: &Args, tf32_override: Option<&str>) -> Result<(), Failure> {
        if args.cost_conv_default {
            require(
                args.mode == "capture"
                    && args.cost_same_seed_context
                    && args.values.get("--cudnn-conv-search").map(String::as_str)
                        == Some("default-device-warm-v1")
                    && tf32_override == Some("0"),
                "arguments",
                "default_cost_requires_explicit_conv_default_and_tf32_override_zero",
            )?;
        }
        Ok(())
    }
    fn pinned(path: &Path, expected: &str, maximum: usize) -> Result<Vec<u8>, Failure> {
        let bytes =
            asset::read_bounded(path, maximum).map_err(|_| fail("input", "bounded_read_failed"))?;
        require(
            hex(&Sha256::digest(&bytes)) == expected,
            "input",
            "digest_mismatch",
        )?;
        Ok(bytes)
    }
    /// Both profiles use the actual public scheduler. The queue profile admits
    /// Fresh WDL comparisons while own CPU traversal can remain incomplete.
    /// Selecting either profile proves no interruption or resume by itself.
    #[derive(Clone, Copy, Debug, Default, Deserialize)]
    #[serde(rename_all = "snake_case")]
    enum ExecutionProfile {
        #[default]
        ActualOpponentContinuationV1,
        IterativeFrozenModelWdlV2,
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Scenario {
        schema: String,
        fen: String,
        /// Exact legal history after the FEN. A FEN alone never invents history.
        moves: Vec<String>,
        line_plies: usize,
        max_rounds: u64,
        cpu_depth: u16,
        max_cpu_nodes: u64,
        cpu_nodes_per_task: u64,
        #[serde(default)]
        execution_profile: ExecutionProfile,
        #[serde(default = "default_role_forwards")]
        max_role_forwards: u64,
    }
    fn default_role_forwards() -> u64 {
        DEFAULT_ROLE_FORWARDS
    }
    impl Scenario {
        fn position(&self) -> Result<Position, Failure> {
            require(
                self.schema == SCENARIO_SCHEMA
                    && self.fen.len() <= 256
                    && self.moves.len() <= 256
                    && (4..=6).contains(&self.line_plies)
                    && (1..=2).contains(&self.max_rounds)
                    && (1..=4).contains(&self.cpu_depth)
                    && (1..=100_000).contains(&self.max_cpu_nodes)
                    && (1..=16_384).contains(&self.cpu_nodes_per_task),
                "scenario",
                "bounds_or_schema_invalid",
            )?;
            require(
                (MIN_ROLE_FORWARDS..=MAX_ROLE_FORWARDS).contains(&self.max_role_forwards),
                "scenario",
                "role_forward_budget_out_of_bounds",
            )?;
            let mut position =
                Position::from_fen(&self.fen).map_err(|_| fail("scenario", "invalid_fen"))?;
            for text in &self.moves {
                let movement = BoardMove::from_uci(text)
                    .map_err(|_| fail("scenario", "invalid_history_move"))?;
                position
                    .make_move(movement)
                    .map_err(|_| fail("scenario", "illegal_history_move"))?;
            }
            require(
                !position.legal_moves().is_empty(),
                "scenario",
                "nonterminal_legal_root_required",
            )?;
            Ok(position)
        }
    }

    #[derive(Clone)]
    struct OwnedQuestion {
        position: Position,
        legal: Vec<BoardMove>,
        prefix: Vec<BoardMove>,
        proposal: Vec<BoardMove>,
        counterexample: Option<Vec<BoardMove>>,
        records: Vec<RoleRecord>,
        context: RoleLogicalContext,
        deadline: Instant,
    }
    impl OwnedQuestion {
        fn copy(query: &RoleQuery<'_>, context: &RoleLogicalContext) -> Self {
            Self {
                position: query.position.clone(),
                legal: query.legal.to_vec(),
                prefix: query.prefix.to_vec(),
                proposal: query.proposal.to_vec(),
                counterexample: query.counterexample.map(<[BoardMove]>::to_vec),
                records: query.records.to_vec(),
                context: context.clone(),
                deadline: query.deadline,
            }
        }
        fn query<'a>(&'a self, cancel: &'a AtomicBool) -> RoleQuery<'a> {
            RoleQuery {
                position: &self.position,
                legal: &self.legal,
                prefix: &self.prefix,
                proposal: &self.proposal,
                counterexample: self.counterexample.as_deref(),
                records: &self.records,
                revision: self.context.public_revision,
                deadline: self.deadline,
                cancel,
            }
        }
    }
    #[derive(Clone)]
    struct FixedCostRepair {
        question: OwnedQuestion,
        input: PalsModelInput,
        source: Value,
        source_latent: Vec<u32>,
        configuration: PalsModelConfig,
        model_epoch: [u8; 32],
    }
    struct CallCapture {
        id: RequestId,
        wire: Value,
        question: Option<OwnedQuestion>,
        snapshot: PositionSnapshot,
        deadline: Instant,
        cancel_address: usize,
        raw: Option<PalsRawOutput>,
        accepted: bool,
    }
    struct CpuAttemptCapture {
        state: PositionIdentity,
        deadline: Option<Instant>,
        cancel_address: usize,
        wire: Value,
    }
    fn paused_stack_spans_repair_and_c_tail(
        paused_at: u64,
        repair_at: u64,
        critic_at: u64,
        resumed_at: u64,
    ) -> bool {
        paused_at < repair_at && repair_at < critic_at && critic_at < resumed_at
    }
    fn resident_repair_request<'a>(calls: impl Iterator<Item = &'a Value>) -> Option<RequestId> {
        // The native bank has one slot per role. Historical accepted Repair
        // packets do not retain a seed after any later accepted P publication.
        // C and ValueFresh never replace that P slot.
        let latest = calls
            .filter(|call| {
                call["accepted"] == true
                    && call["input"]["role"] == "proposer"
                    && matches!(
                        call["logical_context"]["purpose"].as_str(),
                        Some("ProposePolicy" | "RepairPolicy")
                    )
            })
            .last()?;
        if latest["logical_context"]["purpose"] != "RepairPolicy"
            || latest["stage"] != "actual_pals_search"
            || latest["accepted_with_original_controls"] != true
            || latest["physical"]["ready"] != true
            || latest["physical"]["completed_ok"] != true
            || latest["physical"]["completion_unknown"] != false
        {
            return None;
        }
        Some(RequestId {
            epoch: rz_contracts::ProcessEpoch(latest["request_id"]["epoch"].as_u64()?),
            sequence: latest["request_id"]["sequence"].as_u64()?,
        })
    }
    fn resident_repair_for_progress<'a>(
        calls: impl Iterator<Item = &'a Value>,
        cpu_resume: &Value,
    ) -> Option<RequestId> {
        if cpu_resume["passed"] != true {
            return None;
        }
        let resident = resident_repair_request(calls)?;
        cpu_resume["matched_actual_pause_repair_c_resume_sequences"]
            .as_array()?
            .iter()
            .any(|chain| chain["accepted_gpu_repair_request_id"] == request_id(resident))
            .then_some(resident)
    }
    struct Capture {
        configuration: PalsModelConfig,
        started: Instant,
        stage: &'static str,
        calls: Vec<CallCapture>,
        rechecks: Vec<Value>,
        recheck_completed: bool,
        cpu_attempts: Vec<CpuAttemptCapture>,
        cpu_owner_actions: Vec<Value>,
        cpu_trace_overflow: bool,
        scheduler_resume_links: Vec<Value>,
        scheduler_resume_link_overflow: bool,
        progress_replay_source: Option<RequestId>,
        max_role_forwards: u64,
        call_output_bytes_reserved: usize,
        output_reservation_exhausted: bool,
        cost_same_seed_context: bool,
        cost_fixed_repair: Option<FixedCostRepair>,
        cost_warm_lane: Option<Value>,
    }
    impl Capture {
        fn new(started: Instant) -> Self {
            Self {
                configuration: PalsModelConfig::full_line_interaction_v2(),
                started,
                stage: "loading",
                calls: Vec::with_capacity((DEFAULT_ROLE_FORWARDS - STARTUP_ROLE_FORWARDS) as usize),
                rechecks: Vec::with_capacity(16),
                recheck_completed: false,
                cpu_attempts: Vec::with_capacity(MAX_CPU_TRACE_ATTEMPTS),
                cpu_owner_actions: Vec::with_capacity(16),
                cpu_trace_overflow: false,
                scheduler_resume_links: Vec::new(),
                scheduler_resume_link_overflow: false,
                progress_replay_source: None,
                max_role_forwards: DEFAULT_ROLE_FORWARDS,
                call_output_bytes_reserved: 0,
                output_reservation_exhausted: false,
                cost_same_seed_context: false,
                cost_fixed_repair: None,
                cost_warm_lane: None,
            }
        }
        fn call(&mut self, id: RequestId) -> Result<&mut CallCapture, RoleError> {
            self.calls
                .iter_mut()
                .find(|call| call.id == id)
                .ok_or(RoleError::InvalidOutput)
        }
        fn last_repair(&self) -> Option<(OwnedQuestion, PalsModelInput, RequestId, Vec<u32>)> {
            let id = self.calls.iter().rev().find_map(|call| {
                (call.accepted
                    && call.question.as_ref()?.context.purpose == RoleQueryPurpose::RepairPolicy)
                    .then_some(call.id)
            })?;
            self.repair_by_request(id)
        }
        fn repair_by_request(
            &self,
            id: RequestId,
        ) -> Option<(OwnedQuestion, PalsModelInput, RequestId, Vec<u32>)> {
            let call = self.calls.iter().find(|call| call.id == id)?;
            let question = call.question.as_ref()?;
            if !call.accepted || question.context.purpose != RoleQueryPurpose::RepairPolicy {
                return None;
            }
            Some((
                question.clone(),
                serde_json::from_value(call.wire["input"].clone()).ok()?,
                call.id,
                call.raw
                    .as_ref()?
                    .private_latent
                    .iter()
                    .map(|value| value.to_bits())
                    .collect(),
            ))
        }
        fn wires(&self) -> Vec<Value> {
            self.calls.iter().map(|call| call.wire.clone()).collect()
        }
        fn original_pause_source(&self, resume_index: usize) -> Option<usize> {
            let resumed = self.cpu_attempts.get(resume_index)?;
            let mut cursor = resume_index;
            for _ in 0..MAX_CPU_TRACE_ATTEMPTS {
                let current = self.cpu_attempts.get(cursor)?;
                let source_index = current.wire["resume_source_attempt_index"].as_u64()? as usize;
                if source_index >= cursor {
                    return None;
                }
                let source = self.cpu_attempts.get(source_index)?;
                if source.state != resumed.state
                    || source.deadline != resumed.deadline
                    || source.cancel_address != resumed.cancel_address
                    || source.wire["stage"] != "actual_pals_search"
                    || source.wire["returned_ok"] != true
                    || source.wire["report"]["resume_token"]["is_paused_stack"] != true
                    || source.wire["report"]["resume_token"]["owner_current_after"] != true
                    || source.wire["requested"]["max_depth"]
                        != resumed.wire["requested"]["max_depth"]
                {
                    return None;
                }
                if source.wire["kind"] == "analyze" {
                    return Some(source_index);
                }
                if source.wire["kind"] != "resume"
                    || source.wire["resume_token"]["owner_current_before"] != true
                    || source.wire["input_token_current_after"] != false
                {
                    return None;
                }
                cursor = source_index;
            }
            None
        }
        fn cpu_resume_evidence(&self) -> Value {
            self.cpu_resume_evidence_for_repair(self.progress_replay_source)
        }
        fn cpu_resume_evidence_for_repair(&self, required_repair: Option<RequestId>) -> Value {
            let mut matches = Vec::new();
            'attempts: for (resume_index, attempt) in self.cpu_attempts.iter().enumerate() {
                let wire = &attempt.wire;
                if wire["kind"] != "resume"
                    || wire["stage"] != "actual_pals_search"
                    || wire["resume_token"]["is_paused_stack"] != true
                    || wire["resume_token"]["owner_current_before"] != true
                    || wire["input_token_current_after"] != false
                    || wire["returned_ok"] != true
                    || wire["report"]["nodes"].as_u64().unwrap_or(0) == 0
                {
                    continue;
                }
                let Some(source_index) = self.original_pause_source(resume_index) else {
                    continue;
                };
                let Some(source) = self.cpu_attempts.get(source_index) else {
                    continue;
                };
                if source.wire["kind"] != "analyze"
                    || source.wire["stage"] != "actual_pals_search"
                    || source.wire["report"]["resume_token"]["is_paused_stack"] != true
                    || source.wire["report"]["resume_token"]["owner_current_after"] != true
                    || source.wire["returned_ok"] != true
                    || source.state != attempt.state
                    || source.deadline != attempt.deadline
                    || source.cancel_address != attempt.cancel_address
                    || source.wire["requested"]["max_depth"] != wire["requested"]["max_depth"]
                    || source.wire["report"]["completed_depth"].as_u64()
                        >= source.wire["requested"]["max_depth"].as_u64()
                    || wire["resume_token"]["retained_bytes"].as_u64().unwrap_or(0) == 0
                    || wire["resume_token"]["cumulative_work"]["nodes"]
                        .as_u64()
                        .unwrap_or(0)
                        == 0
                {
                    continue;
                }
                let (Some(paused_at), Some(resumed_at)) = (
                    source.wire["finished_elapsed_ns"].as_u64(),
                    wire["started_elapsed_ns"].as_u64(),
                ) else {
                    continue;
                };
                let repairs: Vec<_> = self
                    .calls
                    .iter()
                    .filter(|call| {
                        call.accepted
                            && required_repair.is_none_or(|id| call.id == id)
                            && call.wire["stage"] == "actual_pals_search"
                            && call.question.as_ref().is_some_and(|question| {
                                question.context.purpose == RoleQueryPurpose::RepairPolicy
                            })
                            && call.wire["accepted_context_elapsed_ns"]
                                .as_u64()
                                .is_some_and(|at| paused_at < at && at < resumed_at)
                    })
                    .collect();
                for repair in repairs {
                    let repair_at = repair.wire["accepted_context_elapsed_ns"].as_u64().unwrap();
                    for recheck in &self.rechecks {
                        if recheck["event"] != "finished"
                            || recheck["actual_search_continuation_completed"] != true
                        {
                            continue;
                        }
                        let Some(question) = repair.question.as_ref() else {
                            continue;
                        };
                        let prefix: Vec<_> = question
                            .prefix
                            .iter()
                            .map(|movement| json!(movement.to_string()))
                            .collect();
                        let Some(repaired) = recheck["repaired"].as_array() else {
                            continue;
                        };
                        if repaired.len() <= prefix.len()
                            || !repaired.starts_with(&prefix)
                            || recheck["refutation"]
                                != json!(question.counterexample.as_ref().map(|line| moves(line)))
                            || recheck["reply_context"]["game_generation"]
                                != question.context.game_generation
                            || recheck["reply_context"]["search_generation"]
                                != question.context.search_generation
                        {
                            continue;
                        }
                        let Some(ids) = recheck["actual_new_prefix_c_tail_request_ids"].as_array()
                        else {
                            continue;
                        };
                        let Some(critic) = self.calls.iter().find(|call| {
                            call.accepted
                                && call.wire["stage"] == "actual_pals_search"
                                && ids.contains(&request_id(call.id))
                                && call.wire["accepted_context_elapsed_ns"]
                                    .as_u64()
                                    .is_some_and(|at| {
                                        paused_stack_spans_repair_and_c_tail(
                                            paused_at, repair_at, at, resumed_at,
                                        )
                                    })
                        }) else {
                            continue;
                        };
                        let task_links: Vec<_> = self
                            .scheduler_resume_links
                            .iter()
                            .filter(|link| {
                                link["same_exact_task_key"] == true
                                    && link["actual_resume_attempt_indices"]
                                        .as_array()
                                        .is_some_and(|indices| {
                                            indices.contains(&json!(resume_index))
                                        })
                            })
                            .cloned()
                            .collect();
                        if task_links.is_empty() {
                            continue;
                        }
                        matches.push(json!({
                            "paused_attempt_index": source_index, "resume_attempt_index": resume_index,
                            "accepted_gpu_repair_request_id": request_id(repair.id),
                            "accepted_gpu_c_tail_request_id": request_id(critic.id),
                            "state_and_full_history_exactly_matched": true,
                            "original_deadline_and_cancel_owner_matched": true,
                            "live_paused_stack_owner_checked_before_resume": true,
                            "input_token_consumed_by_actual_cpu_owner": true,
                            "scheduler_task_resume_links": task_links,
                            "new_work": wire["report"]["nodes"],
                            "cpu_pause_preceded_gpu_repair": paused_at < repair_at,
                            "cpu_pause_preceded_gpu_c_tail": paused_at < critic.wire["accepted_context_elapsed_ns"].as_u64().unwrap(),
                            "cpu_pause_finished_elapsed_ns": paused_at, "cpu_resume_started_elapsed_ns": resumed_at,
                            "returned_complete_requested_depth": wire["report"]["completed_depth"]
                                .as_u64() >= wire["requested"]["max_depth"].as_u64(),
                        }));
                        // Search every eligible sequence for the exact resident
                        // Repair before stopping. A different historical chain
                        // cannot witness the single resident P seed.
                        // One concrete matching chain is enough for this gate.
                        // All underlying CPU calls and NN packets remain in the
                        // trace; duplicated proof chains need no output copies.
                        break 'attempts;
                    }
                }
            }
            let observed = !matches.is_empty()
                && !self.cpu_trace_overflow
                && !self.scheduler_resume_link_overflow;
            json!({
                "status": if observed { "observed" } else { "not_exercised" },
                "required_for_gpu_acceptance": true, "passed": observed,
                "paused_stack_selection_is_resume_evidence": false,
                "trace_overflow": self.cpu_trace_overflow,
                "scheduler_link_overflow": self.scheduler_resume_link_overflow,
                "max_trace_attempts": MAX_CPU_TRACE_ATTEMPTS,
                "observed_calls": self.cpu_attempts.iter().map(|call| call.wire.clone()).collect::<Vec<_>>(),
                "actual_owner_actions": self.cpu_owner_actions,
                "scheduler_resume_links": self.scheduler_resume_links,
                "required_accepted_repair_request_id": required_repair.map(request_id),
                "matched_actual_pause_repair_c_resume_sequences": matches,
            })
        }
    }

    fn cpu_work(work: CpuWork) -> Value {
        json!({"nodes": work.nodes, "quiescence_nodes": work.quiescence_nodes, "tt_hits": work.tt_hits})
    }
    fn token_digest(token: &CpuResumeToken) -> String {
        // A diagnostic correlation pin for this compiled implementation. Exact
        // Rules/history comparison and live owner checks grant resume evidence.
        hex(&Sha256::digest(format!("{token:?}").as_bytes()))
    }
    struct ObservedCpu {
        engine: CpuEngine,
        capture: Arc<Mutex<Capture>>,
    }
    impl ObservedCpu {
        fn begin(
            &self,
            kind: &'static str,
            position: &Position,
            line: &[BoardMove],
            limits: CpuLimits,
            cancel: &AtomicBool,
            token: Option<&CpuResumeToken>,
        ) -> Option<usize> {
            let mut capture = self.capture.lock().ok()?;
            if capture.cpu_attempts.len() >= MAX_CPU_TRACE_ATTEMPTS {
                capture.cpu_trace_overflow = true;
                return None;
            }
            let index = capture.cpu_attempts.len();
            let digest = token.map(token_digest);
            let source = digest.as_ref().and_then(|digest| {
                capture.cpu_attempts.iter().rposition(|attempt| {
                    attempt.wire["report"]["resume_token"]["digest_hex"] == *digest
                })
            });
            let token_wire = token.map(|token| {
                json!({
                    "digest_hex": digest, "is_paused_stack": token.is_paused_stack(),
                    "owner_current_before": self.engine.token_is_current(token),
                    "retained_bytes": token.paused_stack_bytes(),
                    "cumulative_work": token.cumulative_work().map(cpu_work),
                })
            });
            let wire = json!({
                "attempt_index": index, "kind": kind, "stage": capture.stage,
                "started_elapsed_ns": capture.started.elapsed().as_nanos(),
                "root_fen": position.to_fen(), "root_and_history_debug_sha256_hex":
                    hex(&Sha256::digest(format!("{:?}", position.position_identity()).as_bytes())),
                "line": moves(line), "requested": {"max_depth": limits.max_depth, "max_nodes": limits.max_nodes,
                    "deadline_elapsed_ns": limits.deadline.map(|at| at.saturating_duration_since(capture.started).as_nanos())},
                "resume_token": token_wire, "resume_source_attempt_index": source,
                "cpu_search_identity": self.engine.search_identity(),
                "cpu_search_conditions": self.engine.search_conditions(),
                "cpu_value_identity": self.engine.value_identity(),
                "paused_stack_bytes_before": self.engine.paused_stack_bytes(),
                "cancel_observed_before": cancel.load(Ordering::Acquire),
                "returned_ok": false,
            });
            capture.cpu_attempts.push(CpuAttemptCapture {
                state: position.position_identity(),
                deadline: limits.deadline,
                cancel_address: cancel as *const AtomicBool as usize,
                wire,
            });
            Some(index)
        }
        fn finish(
            &self,
            index: Option<usize>,
            result: &Result<CpuReport, CpuError>,
            input: Option<&CpuResumeToken>,
        ) {
            let Some(index) = index else {
                return;
            };
            let Ok(mut capture) = self.capture.lock() else {
                return;
            };
            let finished = capture.started.elapsed().as_nanos();
            let Some(attempt) = capture.cpu_attempts.get_mut(index) else {
                return;
            };
            attempt.wire["finished_elapsed_ns"] = json!(finished);
            attempt.wire["last_attempt_work"] =
                json!(self.engine.last_attempt_work().map(cpu_work));
            attempt.wire["paused_stack_bytes_after"] = json!(self.engine.paused_stack_bytes());
            attempt.wire["input_token_current_after"] =
                json!(input.map(|token| self.engine.token_is_current(token)));
            attempt.wire["returned_ok"] = json!(result.is_ok());
            match result {
                Ok(report) => {
                    attempt.wire["report"] = json!({
                        "best_move": report.best_move.map(|movement| movement.to_string()), "pv": moves(&report.pv),
                        "score": report.score, "completed_depth": report.completed_depth, "nodes": report.nodes,
                        "quiescence_nodes": report.quiescence_nodes, "tt_hits": report.tt_hits,
                        "completion": format!("{:?}", report.completion), "score_scope": format!("{:?}", report.score_scope),
                        "profile": format!("{:?}", report.profile), "score_provenance": report.score_provenance,
                        "value_identity": report.value_identity, "search_version": report.search_version,
                        "root_restricted": report.root_restricted, "elapsed_ns": report.elapsed.as_nanos(),
                        "reused_completed_depth": report.reused_completed_depth,
                        "resume_token": report.resume.as_ref().map(|token| json!({
                            "digest_hex": token_digest(token), "is_paused_stack": token.is_paused_stack(),
                            "owner_current_after": self.engine.token_is_current(token),
                            "retained_bytes": token.paused_stack_bytes(),
                            "cumulative_work": token.cumulative_work().map(cpu_work),
                        })),
                    });
                }
                Err(error) => attempt.wire["error"] = json!(format!("{error:?}")),
            }
        }
        fn owner_action(
            &self,
            action: &'static str,
            before: Option<usize>,
            actually_dropped: bool,
        ) {
            let Ok(mut capture) = self.capture.lock() else {
                return;
            };
            if capture.cpu_owner_actions.len() >= MAX_CPU_TRACE_ATTEMPTS {
                capture.cpu_trace_overflow = true;
                return;
            }
            let elapsed = capture.started.elapsed().as_nanos();
            let stage = capture.stage;
            capture.cpu_owner_actions.push(json!({"action": action, "stage": stage,
                "elapsed_ns": elapsed, "paused_stack_bytes_before": before,
                "paused_stack_bytes_after": self.engine.paused_stack_bytes(), "actually_dropped": actually_dropped}));
        }
    }
    // This adapter observes one actual CPU owner. It never runs an additional
    // search, changes an engine limit/report, retains private frames, or invents
    // a scheduler resume. All traversal ownership stays inside CpuEngine.
    impl CpuSearcher for ObservedCpu {
        fn config(&self) -> &CpuConfig {
            self.engine.config()
        }
        fn value_identity(&self) -> &CpuValueIdentity {
            self.engine.value_identity()
        }
        fn search_identity(&self) -> &'static str {
            self.engine.search_identity()
        }
        fn search_conditions(&self) -> String {
            self.engine.search_conditions()
        }
        fn capabilities(&self) -> CpuCapabilities {
            CpuSearcher::capabilities(&self.engine)
        }
        fn resume_policy(&self) -> CpuResumePolicy {
            self.engine.resume_policy()
        }
        fn preserves_paused_on_fresh_tasks(&self) -> bool {
            self.engine.preserves_paused_on_fresh_tasks()
        }
        fn paused_side_scope_snapshot(&self) -> Option<rz_search::cpu::CpuSideScopeSnapshot> {
            self.engine.paused_side_scope_snapshot()
        }
        fn paused_stack_bytes(&self) -> Option<usize> {
            self.engine.paused_stack_bytes()
        }
        fn paused_stack_snapshot(&self) -> Option<CpuPausedStackSnapshot> {
            self.engine.paused_stack_snapshot()
        }
        fn token_is_current(&self, token: &CpuResumeToken) -> bool {
            self.engine.token_is_current(token)
        }
        fn supports_paused_stack_discard(&self) -> bool {
            self.engine.supports_paused_stack_discard()
        }
        fn discard_paused_stack(&mut self) -> bool {
            let before = self.engine.paused_stack_bytes();
            let dropped = self.engine.discard_paused_stack();
            self.owner_action("discard_paused_stack", before, dropped);
            dropped
        }
        fn last_attempt_work(&self) -> Option<CpuWork> {
            self.engine.last_attempt_work()
        }
        fn clear(&mut self) {
            let before = self.engine.paused_stack_bytes();
            self.engine.clear();
            self.owner_action("clear", before, before.is_some());
        }
        fn analyze(
            &mut self,
            position: &Position,
            limits: CpuLimits,
            cancel: &AtomicBool,
        ) -> Result<CpuReport, CpuError> {
            let index = self.begin("analyze", position, &[], limits, cancel, None);
            let result = self.engine.analyze(position, limits, cancel);
            self.finish(index, &result, None);
            result
        }
        fn analyze_root_moves(
            &mut self,
            position: &Position,
            line: &[BoardMove],
            limits: CpuLimits,
            cancel: &AtomicBool,
        ) -> Result<CpuReport, CpuError> {
            let index = self.begin("analyze_root_moves", position, line, limits, cancel, None);
            let result = self
                .engine
                .analyze_root_moves(position, line, limits, cancel);
            self.finish(index, &result, None);
            result
        }
        fn analyze_divergence(
            &mut self,
            position: &Position,
            line: &[BoardMove],
            limits: CpuLimits,
            cancel: &AtomicBool,
        ) -> Result<CpuReport, CpuError> {
            let index = self.begin("analyze_divergence", position, line, limits, cancel, None);
            let result = self
                .engine
                .analyze_divergence(position, line, limits, cancel);
            self.finish(index, &result, None);
            result
        }
        fn resume(
            &mut self,
            position: &Position,
            token: &CpuResumeToken,
            limits: CpuLimits,
            cancel: &AtomicBool,
        ) -> Result<CpuReport, CpuError> {
            let index = self.begin("resume", position, &[], limits, cancel, Some(token));
            let result = self.engine.resume(position, token, limits, cancel);
            self.finish(index, &result, Some(token));
            result
        }
    }

    fn observe_scheduler_resume_links(
        capture: &mut Capture,
        stores: &PalsStores,
    ) -> Result<(), Failure> {
        // This example creates one fresh, non-archive engine. Task IDs are
        // contiguous for that search; absence fails instead of fabricating IDs.
        if stores.tasks.len() > MAX_CPU_TRACE_ATTEMPTS {
            capture.scheduler_resume_link_overflow = true;
            return Ok(());
        }
        for index in 0..stores.tasks.len() {
            let execution = ExecutionId(index);
            let task = stores
                .tasks
                .get(execution)
                .map_err(|_| fail("cpu_trace", "task_record_unavailable"))?;
            let Some(previous) = task.resumed_from else {
                continue;
            };
            if capture
                .scheduler_resume_links
                .iter()
                .any(|link| link["execution"] == json!(execution))
            {
                continue;
            }
            let source = stores
                .tasks
                .get(previous)
                .map_err(|_| fail("cpu_trace", "resume_source_task_unavailable"))?;
            let state = stores
                .states
                .get(task.key.state)
                .map_err(|_| fail("cpu_trace", "resume_rules_state_unavailable"))?;
            let matching: Vec<_> = capture
                .cpu_attempts
                .iter()
                .enumerate()
                .filter_map(|(index, attempt)| {
                    (attempt.wire["kind"] == "resume"
                        && task.key.question == TaskQuestion::AnalyzePosition
                        && attempt.state == state.position_identity()
                        && attempt.wire["requested"]["max_depth"] == task.key.requested_depth
                        && attempt.wire["requested"]["max_nodes"] == task.key.node_budget)
                        .then_some(index)
                })
                .collect();
            capture.scheduler_resume_links.push(json!({"execution": execution, "resumed_from": previous,
                "task_key": task.key, "status": task.status,
                "same_exact_task_key": task.key == source.key, "actual_resume_attempt_indices": matching}));
        }
        Ok(())
    }
    #[derive(Clone)]
    struct Observer(Arc<Mutex<Capture>>);
    fn observe_cost_elapsed(wire: &mut Value, enabled: bool, event: &str, elapsed: Duration) {
        if enabled {
            wire["cost_observation"][event] = json!(elapsed.as_nanos());
        }
    }
    impl NativeRoleObserver for Observer {
        fn prepared(
            &mut self,
            _id: RequestId,
            _input: &PalsModelInput,
            _context: NativePreparedContext<'_>,
        ) -> Result<(), RoleError> {
            Err(RoleError::Backend(
                "CUDA Warm harness requires explicit private observation".into(),
            ))
        }
        fn prepared_private_warm(
            &mut self,
            id: RequestId,
            input: &PalsModelInput,
            context: NativePreparedContext<'_>,
            logical_context: Option<&RoleLogicalContext>,
            prepared: NativePrivateWarmPrepared<'_>,
        ) -> Result<(), RoleError> {
            let mut capture = self.0.lock().map_err(|_| RoleError::Unavailable)?;
            if capture.calls.len() >= (capture.max_role_forwards - STARTUP_ROLE_FORWARDS) as usize
                || capture.calls.iter().any(|call| call.id == id)
                || prepared.initial_latent_bits.len() != LATENT_ELEMENTS
                || prepared
                    .initial_latent_bits
                    .iter()
                    .any(|bits| !f32::from_bits(*bits).is_finite())
                || matches!(prepared.invocation, PrivateInvocation::ApproxWarmV1 { .. })
            {
                return Err(RoleError::InvalidOutput);
            }
            let warm = matches!(
                prepared.invocation,
                PrivateInvocation::ApproxCudaWarmV2 { .. }
            );
            if warm != prepared.seed_provenance.is_some() {
                return Err(RoleError::InvalidOutput);
            }
            let key = input
                .canonical_input_key(&capture.configuration)
                .map_err(|_| RoleError::InvalidOutput)?;
            let invocation_input = match prepared.invocation {
                PrivateInvocation::Fresh { input_key }
                | PrivateInvocation::ApproxCudaWarmV2 { input_key, .. }
                | PrivateInvocation::ApproxWarmV1 { input_key, .. } => input_key,
            };
            if invocation_input != key {
                return Err(RoleError::InvalidOutput);
            }
            let question = match (context, logical_context) {
                (NativePreparedContext::Role { query, .. }, Some(logical)) => {
                    Some(OwnedQuestion::copy(query, logical))
                }
                _ => None,
            };
            let (snapshot, deadline, cancel_address) = match context {
                NativePreparedContext::Role { query, .. } => (
                    query.position.snapshot(),
                    query.deadline,
                    query.cancel as *const AtomicBool as usize,
                ),
                NativePreparedContext::Divergence { query } => (
                    query.root.snapshot(),
                    query.deadline,
                    query.cancel as *const AtomicBool as usize,
                ),
            };
            let input_json_utf8 =
                serde_json::to_string(input).map_err(|_| RoleError::InvalidOutput)?;
            let wire = json!({
                "ordinal": capture.calls.len() + 1, "request_id": request_id(id), "execution_id": null,
                "stage": capture.stage, "prepared_elapsed_ns": capture.started.elapsed().as_nanos(),
                "input": input, "input_key_hex": hex(&key),
                "input_json_sha256_hex": hex(&Sha256::digest(input_json_utf8.as_bytes())),
                "input_json_utf8": input_json_utf8, "input_f32_bits": input_f32_bits(input),
                "initial_latent_bits": prepared.initial_latent_bits, "initial_latent_sha256_hex": bits_digest(prepared.initial_latent_bits),
                "warm_start": warm, "invocation": invocation(prepared.invocation),
                "seed_provenance": prepared.seed_provenance.map(provenance), "bank_at_preparation": prepared.bank,
                "logical_context": logical_context.map(logical),
                "physical": {"dispatched": false, "ready": false, "completion_unknown": false, "completed_ok": false},
                "raw_output": null, "raw_output_bits": null, "accepted": false, "delivered": false,
                "decoded_wdl": null, "fallback": false
            });
            // All shape-bounded raw logits, complete latent bits/values, decoded
            // policy, acceptance context and per-call receipts fit this reserve.
            // The exact preparation packet already includes actual seed/input
            // bytes and provenance. Stop before dispatch if preservation cannot
            // fit; never run a request and silently discard its raw evidence.
            let prepared_bytes = serde_json::to_vec(&wire)
                .map_err(|_| RoleError::InvalidOutput)?
                .len();
            let reservation = prepared_bytes
                .checked_add(CALL_OUTPUT_RESERVE)
                .and_then(|bytes| capture.call_output_bytes_reserved.checked_add(bytes));
            if reservation.is_none_or(|bytes| bytes > OUTPUT_LIMIT - REPORT_HEADROOM) {
                capture.output_reservation_exhausted = true;
                return Err(RoleError::Backend(
                    "CUDA Warm capture output reservation exhausted before dispatch".into(),
                ));
            }
            capture.call_output_bytes_reserved = reservation.unwrap();
            capture.calls.push(CallCapture {
                id,
                wire,
                question,
                snapshot,
                deadline,
                cancel_address,
                raw: None,
                accepted: false,
            });
            Ok(())
        }
        fn dispatched(&mut self, binding: NativeRoleExecutionBinding) -> Result<(), RoleError> {
            let mut capture = self.0.lock().map_err(|_| RoleError::Unavailable)?;
            let elapsed = capture.started.elapsed();
            let cost_enabled = capture.cost_same_seed_context;
            let call = capture.call(binding.request)?;
            if call.wire["physical"]["dispatched"] == true {
                return Err(RoleError::InvalidOutput);
            }
            call.wire["execution_id"] =
                json!({"epoch": binding.execution.epoch.0, "sequence": binding.execution.sequence});
            call.wire["physical"]["dispatched"] = json!(true);
            observe_cost_elapsed(
                &mut call.wire,
                cost_enabled,
                "dispatched_elapsed_ns",
                elapsed,
            );
            Ok(())
        }
        fn terminal(
            &mut self,
            binding: NativeRoleExecutionBinding,
            event: NativeRoleTerminal<'_>,
        ) -> Result<(), RoleError> {
            let mut capture = self.0.lock().map_err(|_| RoleError::Unavailable)?;
            let elapsed = capture.started.elapsed();
            let cost_enabled = capture.cost_same_seed_context;
            let call = capture.call(binding.request)?;
            if call.wire["execution_id"]
                != json!({"epoch": binding.execution.epoch.0, "sequence": binding.execution.sequence})
            {
                return Err(RoleError::InvalidOutput);
            }
            match event {
                NativeRoleTerminal::Ready(result) => {
                    call.wire["physical"]["ready"] = json!(true);
                    call.wire["physical"]["completed_ok"] = json!(match result {
                        Ok(PalsNativeResult::EvaluationWithEvidence { outcome, .. }) =>
                            outcome.is_ok(),
                        _ => result.is_ok(),
                    });
                    if cost_enabled {
                        let (delta, complete) = match result {
                            Ok(PalsNativeResult::EvaluationWithEvidence {
                                actual_completed_nn_inputs_delta,
                                complete,
                                ..
                            }) => (*actual_completed_nn_inputs_delta, *complete),
                            _ => (None, false),
                        };
                        call.wire["cost_observation"]["actual_completed_nn_inputs_delta"] =
                            json!(delta);
                        call.wire["cost_observation"]["nn_input_measurement_complete"] =
                            json!(complete);
                    }
                }
                NativeRoleTerminal::CompletionUnknown(reason) => {
                    call.wire["physical"]["completion_unknown"] = json!(true);
                    call.wire["physical"]["unknown_reason"] = json!(format!("{reason:?}"));
                }
            }
            observe_cost_elapsed(&mut call.wire, cost_enabled, "terminal_elapsed_ns", elapsed);
            Ok(())
        }
        fn physically_completed_with_execution(
            &mut self,
            id: RequestId,
            result: Result<&PalsRawOutput, &BackendError>,
            execution: &NativeExecutionReceipt,
        ) -> Result<(), RoleError> {
            let mut capture = self.0.lock().map_err(|_| RoleError::Unavailable)?;
            let configuration = capture.configuration.clone();
            let elapsed = capture.started.elapsed();
            let cost_enabled = capture.cost_same_seed_context;
            let call = capture.call(id)?;
            observe_cost_elapsed(
                &mut call.wire,
                cost_enabled,
                "physical_callback_elapsed_ns",
                elapsed,
            );
            call.wire["execution"] = json!(execution);
            let Ok(raw) = result else {
                call.wire["raw_failure"] = json!("known_native_failure");
                return Ok(());
            };
            let input: PalsModelInput = serde_json::from_value(call.wire["input"].clone())
                .map_err(|_| RoleError::InvalidOutput)?;
            let decoded = raw
                .decode(&input, &configuration)
                .map_err(|_| RoleError::InvalidOutput)?;
            let bits = json!({
                "candidate_logits": raw.candidate_logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "wdl_logits": raw.wdl_logits.map(f32::to_bits),
                "divergence_logits": raw.divergence_logits.as_ref().map(|v| v.iter().map(|v| v.to_bits()).collect::<Vec<_>>()),
                "task_logits": raw.task_logits.map(|v| v.map(f32::to_bits)),
                "private_latent": raw.private_latent.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            });
            call.wire["raw_output"] = json!(raw);
            call.wire["raw_output_bits"] = bits;
            call.wire["raw_output_json_sha256_hex"] =
                json!(sha_json(raw).map_err(|_| RoleError::InvalidOutput)?);
            call.wire["final_latent_sha256_hex"] = json!(bits_digest(
                &raw.private_latent
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            ));
            call.wire["decoded_wdl"] = json!(decoded.wdl);
            call.wire["decoded_legal_policy"] = json!(decoded.candidate_policy);
            call.raw = Some(raw.clone());
            Ok(())
        }
        fn delivered(&mut self, id: RequestId) -> Result<(), RoleError> {
            self.0
                .lock()
                .map_err(|_| RoleError::Unavailable)?
                .call(id)?
                .wire["delivered"] = json!(true);
            Ok(())
        }
        fn accepted(&mut self, id: RequestId) -> Result<(), RoleError> {
            let mut capture = self.0.lock().map_err(|_| RoleError::Unavailable)?;
            let call = capture.call(id)?;
            if call.accepted
                || call.raw.is_none()
                || call.wire["physical"]["completion_unknown"] == true
            {
                return Err(RoleError::InvalidOutput);
            }
            call.accepted = true;
            call.wire["accepted"] = json!(true);
            Ok(())
        }
        fn accepted_context(
            &mut self,
            id: RequestId,
            acceptance: &RoleAcceptance<'_>,
        ) -> Result<(), RoleError> {
            let mut capture = self.0.lock().map_err(|_| RoleError::Unavailable)?;
            let elapsed = capture.started.elapsed().as_nanos();
            let call = capture.call(id)?;
            if !call.accepted
                || !call.snapshot.same_state(acceptance.snapshot)
                || call.wire["logical_context"] != logical(acceptance.context)
                || call.deadline != acceptance.deadline
                || call.cancel_address != acceptance.cancel as *const AtomicBool as usize
            {
                return Err(RoleError::InvalidOutput);
            }
            call.wire["accepted_context"] = logical(acceptance.context);
            call.wire["accepted_with_original_controls"] = json!(true);
            call.wire["accepted_context_elapsed_ns"] = json!(elapsed);
            Ok(())
        }
        fn rejected(
            &mut self,
            id: RequestId,
            reason: NativeRoleRejection,
        ) -> Result<(), RoleError> {
            self.0
                .lock()
                .map_err(|_| RoleError::Unavailable)?
                .call(id)?
                .wire["rejected"] = json!(format!("{reason:?}"));
            Ok(())
        }
        fn recheck_prepared(&mut self, event: RecheckPrepared<'_>) -> Result<(), RoleError> {
            let mut capture = self.0.lock().map_err(|_| RoleError::Unavailable)?;
            if capture.rechecks.len() >= MAX_RECHECK_EVENTS {
                return Err(RoleError::Unavailable);
            }
            capture.rechecks.push(json!({"event": "prepared", "policy": format!("{:?}", event.policy),
                "identity": format!("{:?}", event.identity), "repair_record_revision": event.repair_record.revision,
                "repaired": moves(event.repaired), "refutation": moves(event.refutation), "anchor_ply": event.anchor_ply,
                "reply_context": logical(event.reply_context), "examined_responses": moves(event.examined_responses),
                "original_deadline_tick": event.deadline_tick}));
            Ok(())
        }
        fn recheck_finished(&mut self, event: RecheckFinished<'_>) -> Result<(), RoleError> {
            let mut capture = self.0.lock().map_err(|_| RoleError::Unavailable)?;
            if capture.rechecks.len() >= MAX_RECHECK_EVENTS {
                return Err(RoleError::Unavailable);
            }
            let identity = format!("{:?}", event.identity);
            let reply_context = logical(event.reply_context);
            let prepared_lines = capture.rechecks.iter().rev().find_map(|prepared| {
                (prepared["event"] == "prepared"
                    && prepared["identity"] == identity
                    && prepared["reply_context"] == reply_context
                    && prepared["original_deadline_tick"] == event.deadline_tick)
                    .then(|| (prepared["repaired"].clone(), prepared["refutation"].clone()))
            });
            let prepared_observation_matched = prepared_lines.is_some();
            let (repaired, refutation) = prepared_lines.unwrap_or((Value::Null, Value::Null));
            let tail_requests: Vec<_> = capture
                .calls
                .iter()
                .filter_map(|call| {
                    let question = call.question.as_ref()?;
                    let context = &question.context;
                    (call.accepted
                        && context.purpose == RoleQueryPurpose::ReplyPolicy
                        && context.proposal_sha256 == event.reply_context.proposal_sha256
                        && context.refutation_sha256 == event.reply_context.refutation_sha256
                        && context.prefix.len() > event.reply_context.prefix.len()
                        && context.prefix.len() < event.counterline.len()
                        && context.prefix == event.counterline[..context.prefix.len()])
                        .then(|| request_id(call.id))
                })
                .collect();
            let completed = matches!(
                event.policy,
                PostRepairRecheckPolicy::ActualOpponentContinuationV1
                    | PostRepairRecheckPolicy::IterativeFrozenModelWdlV2
            ) && prepared_observation_matched
                && event.prepared_accepted
                && event.reply_call_attempted
                && event.reply_accepted
                && event.counterline_completed
                && !event.full_suffix_replayed
                && event.original_error.is_none()
                && !tail_requests.is_empty();
            capture.recheck_completed |= completed;
            capture.rechecks.push(json!({"event": "finished", "policy": format!("{:?}", event.policy),
                "identity": identity, "prepared_accepted": event.prepared_accepted,
                "reply_call_attempted": event.reply_call_attempted, "reply_accepted": event.reply_accepted,
                "selected_response": event.selected_response.map(|v| v.to_string()), "counterline": moves(event.counterline),
                "counterline_completed": event.counterline_completed, "full_suffix_replayed": event.full_suffix_replayed,
                "comparable": event.comparable, "publication_observed": event.publication.is_some(),
                "disposition": format!("{:?}", event.disposition), "original_error": event.original_error.map(|v| format!("{v:?}")),
                "reply_context": reply_context, "original_deadline_tick": event.deadline_tick,
                "repaired": repaired, "refutation": refutation, "prepared_observation_matched": prepared_observation_matched,
                "actual_new_prefix_c_tail_request_ids": tail_requests,
                "actual_search_continuation_completed": completed}));
            Ok(())
        }
    }

    // One shared native owner lets the caller inspect the actual search question
    // after PalsEngine returns. The adapter forwards every output unchanged.
    struct Model {
        native: Arc<Mutex<NativeRoleModel>>,
        identity: String,
        value_identity: ModelValueIdentity,
        forwards: Arc<ForwardBudget>,
    }
    #[derive(Clone, Copy)]
    enum ForwardLane {
        Search,
        Followup,
    }
    struct ForwardBudget {
        // Low/high u32 lanes contain search/followup admission counts. One CAS
        // charges the selected lane and global sum together, including reentry.
        // The registered maximum is at most 128, far below either lane width.
        admitted: AtomicU64,
        maximum: u64,
    }
    impl ForwardBudget {
        fn new(maximum: u64) -> Self {
            Self {
                admitted: AtomicU64::new(0),
                maximum,
            }
        }
        fn usage(&self, ordering: Ordering) -> (u64, u64, u64) {
            let lanes = self.admitted.load(ordering);
            (
                STARTUP_ROLE_FORWARDS,
                lanes & u64::from(u32::MAX),
                lanes >> u32::BITS,
            )
        }
        fn load(&self, ordering: Ordering) -> u64 {
            let (startup, search, followup) = self.usage(ordering);
            startup + search + followup
        }
        fn observation(&self) -> Value {
            let (startup, search, followup) = self.usage(Ordering::Acquire);
            json!({"global_limit":self.maximum,
                "search_limit":self.maximum.checked_sub(STARTUP_ROLE_FORWARDS + FOLLOWUP_ROLE_FORWARDS),
                "followup_limit":FOLLOWUP_ROLE_FORWARDS, "startup_reserved":startup,
                "search_admissions":search, "followup_admissions":followup,
                "global_admissions_including_startup_reservation":startup + search + followup,
                "scope":"admission_charges;_actual_NN_and_completion_use_separate_native_receipts",
                "failed_admission_refunds_or_retries":false})
        }
    }
    fn admission(forwards: &ForwardBudget) -> Result<(), RoleError> {
        forward_admission(forwards, ForwardLane::Followup)
    }
    fn search_admission(forwards: &ForwardBudget) -> Result<(), RoleError> {
        // Search has its own fixed quota. Already charged mid-search Warm and
        // Fresh Value followups must not reserve their two global slots twice.
        forward_admission(forwards, ForwardLane::Search)
    }
    fn forward_admission(forwards: &ForwardBudget, lane: ForwardLane) -> Result<(), RoleError> {
        let search_limit = forwards
            .maximum
            .checked_sub(STARTUP_ROLE_FORWARDS + FOLLOWUP_ROLE_FORWARDS)
            .filter(|_| forwards.maximum <= MAX_ROLE_FORWARDS)
            .ok_or(RoleError::Unavailable)?;
        forwards
            .admitted
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |lanes| {
                let search = lanes & u64::from(u32::MAX);
                let followup = lanes >> u32::BITS;
                if STARTUP_ROLE_FORWARDS + search + followup >= forwards.maximum {
                    return None;
                }
                match lane {
                    ForwardLane::Search if search < search_limit => Some(lanes + 1),
                    ForwardLane::Followup if followup < FOLLOWUP_ROLE_FORWARDS => {
                        Some(lanes + (1u64 << u32::BITS))
                    }
                    _ => None,
                }
            })
            .map(|_| ())
            .map_err(|_| RoleError::Unavailable)
    }
    impl RoleModel for Model {
        fn identity(&self) -> &str {
            &self.identity
        }
        fn value_identity(&self) -> Option<&ModelValueIdentity> {
            Some(&self.value_identity)
        }
        fn propose(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            search_admission(&self.forwards)?;
            self.native
                .lock()
                .map_err(|_| RoleError::Unavailable)?
                .propose(query)
        }
        fn reply(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            search_admission(&self.forwards)?;
            self.native
                .lock()
                .map_err(|_| RoleError::Unavailable)?
                .reply(query)
        }
        fn repair(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            search_admission(&self.forwards)?;
            self.native
                .lock()
                .map_err(|_| RoleError::Unavailable)?
                .repair(query)
        }
        fn divergences(&mut self, query: DivergenceQuery<'_>) -> Result<Vec<f32>, RoleError> {
            search_admission(&self.forwards)?;
            self.native
                .lock()
                .map_err(|_| RoleError::Unavailable)?
                .divergences(query)
        }
        fn propose_with_context(
            &mut self,
            query: RoleQuery<'_>,
            context: &RoleLogicalContext,
        ) -> Result<RoleEvaluation, RoleError> {
            search_admission(&self.forwards)?;
            self.native
                .lock()
                .map_err(|_| RoleError::Unavailable)?
                .propose_with_context(query, context)
        }
        fn reply_with_context(
            &mut self,
            query: RoleQuery<'_>,
            context: &RoleLogicalContext,
        ) -> Result<RoleEvaluation, RoleError> {
            search_admission(&self.forwards)?;
            self.native
                .lock()
                .map_err(|_| RoleError::Unavailable)?
                .reply_with_context(query, context)
        }
        fn repair_with_context(
            &mut self,
            query: RoleQuery<'_>,
            context: &RoleLogicalContext,
        ) -> Result<RoleEvaluation, RoleError> {
            search_admission(&self.forwards)?;
            self.native
                .lock()
                .map_err(|_| RoleError::Unavailable)?
                .repair_with_context(query, context)
        }
        fn divergences_with_context(
            &mut self,
            query: DivergenceQuery<'_>,
            context: &RoleLogicalContext,
        ) -> Result<Vec<f32>, RoleError> {
            search_admission(&self.forwards)?;
            self.native
                .lock()
                .map_err(|_| RoleError::Unavailable)?
                .divergences_with_context(query, context)
        }
        fn evaluate_value(&mut self, query: RoleQuery<'_>) -> Result<ModelValueOutput, RoleError> {
            search_admission(&self.forwards)?;
            self.native
                .lock()
                .map_err(|_| RoleError::Unavailable)?
                .evaluate_value(query)
        }
        fn evaluate_value_with_context(
            &mut self,
            query: RoleQuery<'_>,
            context: &RoleLogicalContext,
        ) -> Result<ModelValueOutput, RoleError> {
            search_admission(&self.forwards)?;
            self.native
                .lock()
                .map_err(|_| RoleError::Unavailable)?
                .evaluate_value_with_context(query, context)
        }
        fn accepted_output_checked(
            &mut self,
            acceptance: RoleAcceptance<'_>,
        ) -> Result<(), RoleError> {
            self.native
                .lock()
                .map_err(|_| RoleError::Unavailable)?
                .accepted_output_checked(acceptance)
        }
        fn recheck_prepared(&mut self, event: RecheckPrepared<'_>) -> Result<(), RoleError> {
            self.native
                .lock()
                .map_err(|_| RoleError::Unavailable)?
                .recheck_prepared(event)
        }
        fn recheck_finished(&mut self, event: RecheckFinished<'_>) -> Result<(), RoleError> {
            self.native
                .lock()
                .map_err(|_| RoleError::Unavailable)?
                .recheck_finished(event)
        }
        fn finish_search(&mut self, reason: RoleSearchClosure) {
            if let Ok(mut native) = self.native.lock() {
                native.finish_search(reason);
            }
        }
        fn new_game_with_generation(&mut self, generation: Option<u64>) {
            if let Ok(mut native) = self.native.lock() {
                native.new_game_with_generation(generation);
            }
        }
    }

    fn same_nonrecord(a: &PalsModelInput, b: &PalsModelInput) -> bool {
        a.role == b.role
            && a.board == b.board
            && a.metadata.map(f32::to_bits) == b.metadata.map(f32::to_bits)
            && a.query.map(f32::to_bits) == b.query.map(f32::to_bits)
            && a.candidates == b.candidates
            && a.divergence_features
                .iter()
                .map(|v| v.map(f32::to_bits))
                .eq(b.divergence_features.iter().map(|v| v.map(f32::to_bits)))
            && a.history_digest == b.history_digest
            && a.model_epoch == b.model_epoch
            && match (&a.full_line, &b.full_line) {
                (Some(a), Some(b)) => {
                    a.query_prefix == b.query_prefix
                        && a.query_proposal == b.query_proposal
                        && a.query_counter == b.query_counter
                }
                (None, None) => true,
                _ => false,
            }
    }
    struct RepairProgress<'a> {
        records: &'a [RoleRecord],
        stores: &'a PalsStores,
        source_request: RequestId,
    }

    fn replay(
        model: &mut NativeRoleModel,
        handle: &NativeRoleFinishHandle,
        capture: &Arc<Mutex<Capture>>,
        forwards: &ForwardBudget,
        progress: RepairProgress<'_>,
        cancel: &AtomicBool,
    ) -> Result<Value, Failure> {
        let RepairProgress {
            records,
            stores,
            source_request,
        } = progress;
        let (original, old_input, source_request, source_latent) = capture
            .lock()
            .map_err(|_| fail("resume", "capture_unavailable"))?
            .repair_by_request(source_request)
            .ok_or(fail("resume", "actual_accepted_search_repair_missing"))?;
        let mut current = original.clone();
        let revision = records.iter().map(|v| v.revision).max().unwrap_or(0);
        require(
            revision > original.context.public_revision,
            "resume",
            "actual_public_record_revision_did_not_advance",
        )?;
        current.records = records.to_vec();
        current.context.public_revision = revision;
        let situation = stores
            .situations
            .get(current.context.situation)
            .map_err(|_| fail("resume", "actual_situation_handle_invalid"))?;
        require(
            situation.state == current.context.state
                && situation.focus == current.context.focus
                && situation.revision == current.context.situation_revision
                && stores
                    .states
                    .get(situation.state)
                    .is_ok_and(|state| state.same_state(&current.position.snapshot())),
            "resume",
            "actual_store_rules_or_focus_changed",
        )?;
        let input = prepare_role_input_for_config(
            &current.query(cancel),
            NativeQueryKind::Repair,
            model.model_epoch(),
            model.model_config(),
        )
        .map_err(|error| role_failure("resume_prepare", error))?;
        require(
            same_nonrecord(&old_input, &input)
                && current.position.position_identity() == original.position.position_identity()
                && current.deadline == original.deadline,
            "resume",
            "nonrecord_rules_history_question_or_control_changed",
        )?;
        let before_key = old_input
            .canonical_input_key(model.model_config())
            .map_err(|_| fail("resume", "input_key_failed"))?;
        let after_key = input
            .canonical_input_key(model.model_config())
            .map_err(|_| fail("resume", "input_key_failed"))?;
        require(
            before_key != after_key
                && old_input.public_memory_key(model.model_config()).ok()
                    != input.public_memory_key(model.model_config()).ok(),
            "resume",
            "actual_record_model_inputs_did_not_change",
        )?;
        let before = handle.receipt();
        let bank_before = before
            .private_warm_observation
            .as_ref()
            .ok_or(fail("resume", "bank_observation_missing"))?;
        capture
            .lock()
            .map_err(|_| fail("resume", "capture_unavailable"))?
            .stage = "same_question_record_revision_resume";
        admission(forwards).map_err(|error| role_failure("resume", error))?;
        let output = model
            .repair_with_context(current.query(cancel), &current.context)
            .map_err(|error| role_failure("resume", error))?;
        let staged = handle.receipt();
        let bank_staged = staged
            .private_warm_observation
            .as_ref()
            .ok_or(fail("resume", "bank_observation_missing"))?;
        require(
            bank_staged.pending_acceptance
                && bank_staged.accepted_seeds == bank_before.accepted_seeds
                && bank_staged.warm_seed_admissions == bank_before.warm_seed_admissions + 1,
            "resume",
            "seed_not_provisional_or_not_warm",
        )?;
        {
            let capture = capture
                .lock()
                .map_err(|_| fail("resume", "capture_unavailable"))?;
            let call = capture
                .calls
                .last()
                .ok_or(fail("resume", "actual_capture_missing"))?;
            let initial: Vec<u32> =
                serde_json::from_value(call.wire["initial_latent_bits"].clone())
                    .map_err(|_| fail("resume", "seed_bits_missing"))?;
            require(
                call.wire["warm_start"] == true
                    && initial == source_latent
                    && call.wire["seed_provenance"]["source_input_hex"] == hex(&before_key),
                "resume",
                "accepted_source_final_latent_not_reused",
            )?;
        }
        model
            .accepted_output_checked(RoleAcceptance {
                snapshot: &current.position.snapshot(),
                context: &current.context,
                deadline: current.deadline,
                cancel,
            })
            .map_err(|error| role_failure("resume_accept", error))?;
        let after = handle.receipt();
        let bank_after = after
            .private_warm_observation
            .as_ref()
            .ok_or(fail("resume", "bank_observation_missing"))?;
        require(
            bank_after.accepted_seeds == bank_before.accepted_seeds + 1
                && !bank_after.pending_acceptance
                && bank_after.pinned_entries == 0
                && after.physical_runs_in_flight == 0
                && !after.quarantined,
            "resume",
            "accepted_seed_or_physical_closure_missing",
        )?;

        let mut value = current.clone();
        value.context.purpose = RoleQueryPurpose::ValueFresh;
        capture
            .lock()
            .map_err(|_| fail("value", "capture_unavailable"))?
            .stage = "value_fresh";
        admission(forwards).map_err(|error| role_failure("value", error))?;
        let value_before = handle.receipt();
        let estimate = model
            .evaluate_value_with_context(value.query(cancel), &value.context)
            .map_err(|error| role_failure("value", error))?;
        model
            .accepted_output_checked(RoleAcceptance {
                snapshot: &value.position.snapshot(),
                context: &value.context,
                deadline: value.deadline,
                cancel,
            })
            .map_err(|error| role_failure("value_accept", error))?;
        let value_after = handle.receipt();
        let capture = capture
            .lock()
            .map_err(|_| fail("value", "capture_unavailable"))?;
        let value_call = capture
            .calls
            .last()
            .ok_or(fail("value", "actual_capture_missing"))?;
        require(
            value_call.wire["warm_start"] == false
                && value_call.wire["invocation"]["mode"] == "fresh"
                && value_after
                    .private_warm_observation
                    .as_ref()
                    .map(|v| v.accepted_seeds)
                    == value_before
                        .private_warm_observation
                        .as_ref()
                        .map(|v| v.accepted_seeds),
            "value",
            "value_used_or_published_seed",
        )?;
        Ok(
            json!({"source_request_id": request_id(source_request), "old_input_key_hex": hex(&before_key),
            "current_input_key_hex": hex(&after_key), "old_record_revision": original.context.public_revision,
            "current_record_revision": revision, "same_exact_nonrecord_and_rules_history": true,
            "same_original_deadline_and_cancel": true, "accepted_full_6144_seed_reused": true,
            "current_store_state_situation_and_focus_checked": true,
            "resume_consumer": "read_only_actual_search_progress_before_next_P_publication",
            "source_resident_at_actual_progress_checked": true,
            "repair_logits": output.logits, "repair_wdl": output.wdl, "value_wdl": estimate.wdl,
            "value_fresh_checked": true, "receipt_before": before, "receipt_staged": staged, "receipt_after": after,
            "receipt_after_value": value_after}),
        )
    }

    fn cost_work_wall(args: &Args) -> Duration {
        if args.cost_same_seed_context {
            if args.cost_conv_default {
                COST_CONV_DEFAULT_WALL
            } else {
                COST_WALL
            }
        } else {
            WALL
        }
    }
    fn cost_counts(receipt: &NativeRoleReceipt) -> Value {
        json!({
            "process_epoch": receipt.process_epoch, "game_generation": receipt.game_generation,
            "request_high_water": receipt.request_high_water,
            "execution_high_water": receipt.execution_high_water,
            "physically_completed_role_calls": receipt.physically_completed_role_calls,
            "completed_role_inputs": receipt.completed_role_inputs,
            "delivered_role_inputs": receipt.delivered_role_inputs,
            "search_consumed_role_inputs": receipt.search_consumed_role_inputs,
            "physical_runs_in_flight": receipt.physical_runs_in_flight,
            "quarantined": receipt.quarantined,
            "private_warm_observation": receipt.private_warm_observation,
            "backend_stats": receipt.backend_stats,
            "backend_stats_observation": receipt.backend_stats_observation,
        })
    }
    fn cost_delta(before: &Value, after: &Value, key: &str) -> Option<u64> {
        let old = before[key].as_u64()?;
        let new = after[key].as_u64()?;
        if old == u64::MAX || new == u64::MAX {
            return None;
        }
        new.checked_sub(old)
    }
    fn cost_known_unconsumed(before: &Value, after: &Value, call: &Value) -> bool {
        cost_delta(before, after, "physically_completed_role_calls") == Some(1)
            && cost_delta(before, after, "completed_role_inputs") == Some(1)
            && cost_delta(before, after, "delivered_role_inputs") == Some(1)
            && cost_delta(before, after, "search_consumed_role_inputs") == Some(0)
            && after["physical_runs_in_flight"] == 0
            && after["quarantined"] == false
            && call["physical"]["dispatched"] == true
            && call["physical"]["ready"] == true
            && call["physical"]["completed_ok"] == true
            && call["physical"]["completion_unknown"] == false
            && call["delivered"] == true
            && call["accepted"] == false
            && call["rejected"] == "SearchFailedUnconsumed"
            && !call["raw_output"].is_null()
            && after["private_warm_observation"]["pending_acceptance"] == false
            && after["private_warm_observation"]["pinned_entries"] == 0
            && after["private_warm_observation"]["active_lease"].is_null()
            && after["private_warm_observation"]["accepted_seeds"]
                .as_u64()
                .is_some()
            && after["private_warm_observation"]["accepted_seeds"]
                == before["private_warm_observation"]["accepted_seeds"]
    }
    fn fixed_cost_packet(
        call: &Value,
        input: &Value,
        context: &Value,
        latent: &[u32],
        source: &Value,
        warm: bool,
        stable_provenance: &mut Option<Value>,
    ) -> bool {
        if context["purpose"] != "RepairPolicy"
            || call["input"] != *input
            || call["logical_context"] != *context
        {
            return false;
        }
        let Ok(initial) = serde_json::from_value::<Vec<u32>>(call["initial_latent_bits"].clone())
        else {
            return false;
        };
        if initial.len() != LATENT_ELEMENTS
            || initial.iter().any(|v| !f32::from_bits(*v).is_finite())
        {
            return false;
        }
        if warm {
            let seed = &call["seed_provenance"];
            if call["warm_start"] != true
                || call["invocation"]["mode"] != "approx_cuda_warm_v2"
                || initial != latent
                || seed.is_null()
                || seed["source_lease_id"].as_u64().is_none_or(|v| v == 0)
                || seed["seed_sequence"].as_u64().is_none_or(|v| v == 0)
                || seed["seal_hex"].as_str().is_none_or(|v| !valid_digest(v))
                || call["invocation"]["seed_seal_hex"] != seed["seal_hex"]
                || seed["source_input_hex"] != source["input_key_hex"]
                || seed["latent_bits_digest_hex"] != seed_provenance_digest(latent)
            {
                return false;
            }
            if let Some(stable) = stable_provenance {
                if seed != &*stable {
                    return false;
                }
            } else {
                *stable_provenance = Some(seed.clone());
            }
            true
        } else {
            call["warm_start"] == false
                && call["invocation"]["mode"] == "fresh"
                && call["seed_provenance"].is_null()
                && initial.iter().all(|value| *value == 0f32.to_bits())
        }
    }
    fn cost_trace(
        started: Instant,
        fixed: &FixedCostRepair,
        reserved: usize,
    ) -> Arc<Mutex<Capture>> {
        let mut trace = Capture::new(started);
        trace.configuration = fixed.configuration.clone();
        trace.cost_same_seed_context = true;
        trace.max_role_forwards = STARTUP_ROLE_FORWARDS + COST_LANE_CALLS as u64;
        // All three streams use the original output limit before dispatch.
        trace.call_output_bytes_reserved = reserved;
        Arc::new(Mutex::new(trace))
    }
    fn run_cost_lane(
        owner: &mut NativeRoleModel,
        handle: &NativeRoleFinishHandle,
        trace: &Arc<Mutex<Capture>>,
        fixed: &FixedCostRepair,
        warm: bool,
        cancel: &AtomicBool,
    ) -> Value {
        let lane = if warm {
            "warm_same_accepted_seed"
        } else {
            "fresh_missing_seed"
        };
        let mut stable_provenance = None;
        let mut preparation_after = None;
        let mut completed_samples = 0;
        let outcome = (|| {
            require(
                fixed.question.context.purpose == RoleQueryPurpose::RepairPolicy
                    && owner.model_epoch() == fixed.model_epoch
                    && sha_json(owner.model_config())? == sha_json(&fixed.configuration)?,
                "cost",
                "repair_model_or_precision_namespace_changed",
            )?;
            owner
                .set_private_warm_observer(Box::new(Observer(Arc::clone(trace))))
                .map_err(|error| role_failure("cost_observer", error))?;
            let expected_input = json!(fixed.input);
            let expected_context = logical(&fixed.question.context);
            for index in 0..COST_LANE_CALLS {
                require(
                    !cancel.load(Ordering::Acquire) && Instant::now() < fixed.question.deadline,
                    "cost",
                    "original_deadline_or_cancel",
                )?;
                let preparation = index < COST_PREPARATION_CALLS;
                let mut trace_guard = trace
                    .lock()
                    .map_err(|_| fail("cost", "capture_unavailable"))?;
                trace_guard.stage = if preparation {
                    "cost_preparation"
                } else {
                    "cost_measurement"
                };
                let started = trace_guard.started;
                let count_before = trace_guard.calls.len();
                drop(trace_guard);
                let before = cost_counts(&handle.receipt());
                let api_started = Instant::now();
                let output = owner
                    .repair_with_context(fixed.question.query(cancel), &fixed.question.context);
                let api_finished = Instant::now();
                let returned = cost_counts(&handle.receipt());
                // No cost output publishes a seed or receives an acceptance.
                // The public closure drops only a known logical output token.
                let closed = owner.close_unconsumed(NativeRoleRejection::SearchFailedUnconsumed);
                let after = cost_counts(&handle.receipt());
                let mut trace_guard = trace
                    .lock()
                    .map_err(|_| fail("cost", "capture_unavailable"))?;
                require(
                    trace_guard.calls.len() == count_before + 1,
                    "cost",
                    "actual_call_missing",
                )?;
                let call = trace_guard
                    .calls
                    .last_mut()
                    .ok_or(fail("cost", "actual_call_missing"))?;
                call.wire["cost_sample"] = json!({
                    "lane": lane, "phase": if preparation { "preparation" } else { "measurement" },
                    "ordinal_in_phase": if preparation { index + 1 } else { index + 1 - COST_PREPARATION_CALLS },
                    "repair_with_context_api_started_elapsed_ns": api_started.duration_since(started).as_nanos(),
                    "repair_with_context_api_finished_elapsed_ns": api_finished.duration_since(started).as_nanos(),
                    "repair_with_context_api_wall_ns": api_finished.duration_since(api_started).as_nanos(),
                    "api_returned_ok": output.is_ok(), "close_unconsumed_returned_ok": closed.is_ok(),
                    "receipt_before": before, "receipt_returned": returned, "receipt_after_close": after,
                    "native_physical_calls_delta": cost_delta(&before, &after, "physically_completed_role_calls"),
                    "native_completed_role_inputs_delta": cost_delta(&before, &after, "completed_role_inputs"),
                    "native_delivered_inputs_delta": cost_delta(&before, &after, "delivered_role_inputs"),
                    "native_search_consumed_inputs_delta": cost_delta(&before, &after, "search_consumed_role_inputs"),
                    "public_cache_preparation_counters": null,
                    "public_cache_preparation_counters_observation": "not_exposed_by_live_native_finish_receipt",
                });
                output.map_err(|error| role_failure("cost_repair", error))?;
                closed.map_err(|error| role_failure("cost_close_unconsumed", error))?;
                require(
                    cost_known_unconsumed(&before, &after, &call.wire)
                        && fixed_cost_packet(
                            &call.wire,
                            &expected_input,
                            &expected_context,
                            &fixed.source_latent,
                            &fixed.source,
                            warm,
                            &mut stable_provenance,
                        ),
                    "cost",
                    "same_input_seed_provenance_or_known_consumer_zero_not_preserved",
                )?;
                if !warm {
                    require(
                        after["private_warm_observation"]["accepted_seeds"] == 0,
                        "cost",
                        "fresh_control_published_seed",
                    )?;
                }
                if index + 1 == COST_PREPARATION_CALLS {
                    preparation_after = Some(after);
                }
                if !preparation {
                    completed_samples += 1;
                }
            }
            Ok::<_, Failure>(())
        })();
        let trace = trace.lock().ok();
        json!({
            "lane": lane, "complete": outcome.is_ok() && completed_samples == COST_MEASURED_CALLS,
            "failure": outcome.err().map(Failure::wire),
            "preparation_calls_requested": COST_PREPARATION_CALLS,
            "measured_calls_requested": COST_MEASURED_CALLS,
            "measured_calls_completed": completed_samples,
            "stable_accepted_seed_provenance": stable_provenance,
            "seed_publication_by_cost_outputs": false,
            "accepted_consumer_scope": "known_physical_output_closed_logically_unconsumed",
            "receipt_after_preparation": preparation_after,
            "preparation_backend_stats": preparation_after.as_ref().and_then(|v| v.get("backend_stats")).filter(|v| !v.is_null()),
            "call_output_bytes_reserved": trace.as_ref().map(|v| v.call_output_bytes_reserved),
            "calls": trace.as_ref().map(|v| v.wires()),
            "live_receipt_after_lane": cost_counts(&handle.receipt()),
        })
    }
    fn prepare_warm_cost(
        owner: &mut NativeRoleModel,
        handle: &NativeRoleFinishHandle,
        capture: &Arc<Mutex<Capture>>,
        started: Instant,
        cancel: &AtomicBool,
    ) -> Result<(FixedCostRepair, Value), Failure> {
        let captured = capture
            .lock()
            .map_err(|_| fail("cost_source", "capture_unavailable"))?;
        let (question, input, request, source_latent) = captured
            .last_repair()
            .ok_or(fail("cost_source", "actual_accepted_repair_missing"))?;
        let source = captured
            .calls
            .iter()
            .find(|call| call.id == request)
            .ok_or(fail("cost_source", "actual_accepted_repair_missing"))?
            .wire
            .clone();
        require(
            question.context.purpose == RoleQueryPurpose::RepairPolicy
                && source["accepted"] == true
                && source["accepted_with_original_controls"] == true
                && source_latent.len() == LATENT_ELEMENTS
                && source_latent.iter().all(|v| f32::from_bits(*v).is_finite()),
            "cost_source",
            "actual_accepted_complete_repair_source_required",
        )?;
        let fixed = FixedCostRepair {
            question,
            input,
            source,
            source_latent,
            configuration: captured.configuration.clone(),
            model_epoch: owner.model_epoch(),
        };
        let integration_inputs_at_progress = captured.calls.len() as u64;
        let receipt_at_resident_progress = cost_counts(&handle.receipt());
        let trace = cost_trace(started, &fixed, captured.call_output_bytes_reserved);
        drop(captured);
        let mut lane = run_cost_lane(owner, handle, &trace, &fixed, true, cancel);
        lane["integration_inputs_at_progress"] = json!(integration_inputs_at_progress);
        lane["receipt_at_resident_progress"] = receipt_at_resident_progress;
        if let Err(error) = owner.set_private_warm_observer(Box::new(Observer(Arc::clone(capture))))
        {
            lane["complete"] = json!(false);
            lane["integration_observer_restore_failure"] =
                role_failure("cost_restore_integration_observer", error).wire();
        }
        Ok((fixed, lane))
    }
    fn run_fresh_cost_owner(
        args: &Args,
        fixed: &FixedCostRepair,
        warm: &Value,
        started: Instant,
        deadline: Instant,
        whole: Instant,
        cancel: &AtomicBool,
    ) -> Result<Value, Failure> {
        cost_default_environment(args, std::env::var("NVIDIA_TF32_OVERRIDE").ok().as_deref())?;
        require(
            !cancel.load(Ordering::Acquire) && Instant::now() < deadline,
            "cost_fresh_load",
            "original_deadline_or_cancel",
        )?;
        let reserved = warm["call_output_bytes_reserved"]
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .ok_or(fail("cost_fresh_load", "output_reservation_unknown"))?;
        let bundle_bytes = pinned(
            &args.path("--cuda-bundle")?,
            args.value("--cuda-bundle-sha256")?,
            64 * 1024,
        )?;
        let spec = CudaRuntimeBundleSpec::from_json(
            std::str::from_utf8(&bundle_bytes)
                .map_err(|_| fail("cost_fresh_load", "bundle_utf8_invalid"))?,
        )
        .map_err(|_| fail("cost_fresh_load", "bundle_invalid"))?;
        let cache = RuntimeCache::open(&args.path("--runtime-output-root")?)
            .map_err(|_| fail("cost_fresh_load", "runtime_cache_unavailable"))?;
        let pin = cache
            .cuda_bundle(&args.path("--runtime-root")?, &spec)
            .map_err(|_| fail("cost_fresh_load", "bundle_pin_failed"))?;
        require(
            Instant::now() < deadline,
            "cost_fresh_load",
            "deadline_after_pin",
        )?;
        let runtime = OrtRuntime::load(&pin)
            .map_err(|_| fail("cost_fresh_load", "full_cuda_bootstrap_failed"))?;
        let control = if args.values.contains_key("--control-inventory") {
            Some(
                PalsCudaControlPolicy::from_inventory(
                    &args.path("--control-inventory")?,
                    args.value("--control-inventory-sha256")?,
                )
                .map_err(|_| fail("cost_fresh_load", "control_inventory_invalid"))?,
            )
        } else {
            None
        };
        // The original profile root was created exclusively by the first owner.
        // The second owner gets a new child, with the same selected policy.
        let profile = args
            .values
            .get("--profile-root")
            .map(|path| PathBuf::from(path).join("cost-fresh-owner"));
        if let Some(profile) = &profile {
            external(profile, true)?;
        }
        let load_started = Instant::now();
        let mut owner = NativeRoleModel::load_with_runtime_and_cuda_private_warm_and_observations(
            &args.path("--export-manifest")?,
            args.value("--export-manifest-sha256")?,
            runtime,
            PalsOnnxConfig {
                provider: Provider::Cuda {
                    device_id: 0,
                    arena_bytes: 256 * 1024 * 1024,
                },
                intra_threads: 1,
                cache_public_memory: true,
                device_public_memory: true,
                cudnn_conv_search: cuda_conv_selection(args)?,
            },
            NativeOwnerOptions {
                drain_limit: CLEANUP,
                host_record_pages: None,
            },
            control.zip(profile.as_deref()),
            Some(args.cuda_warm_limits()?),
        )
        .map_err(|error| role_failure("cost_fresh_load", error))?;
        let handle = owner.finish_handle();
        let source = owner.source_identity();
        let trace = cost_trace(started, fixed, reserved);
        let mut startup = None;
        let outcome = (|| {
            require(
                owner.model_epoch() == fixed.model_epoch
                    && sha_json(owner.model_config())? == sha_json(&fixed.configuration)?,
                "cost_fresh_load",
                "model_or_encoding_namespace_changed",
            )?;
            owner
                .set_private_warm_observer(Box::new(Observer(Arc::clone(&trace))))
                .map_err(|error| role_failure("cost_fresh_observer", error))?;
            owner
                .configure_startup_probe_timeout(Some(30_000))
                .map_err(|error| role_failure("cost_fresh_startup", error))?;
            owner
                .observe_startup_loading(load_started.elapsed())
                .map_err(|error| role_failure("cost_fresh_startup", error))?;
            owner.new_game_with_generation(Some(fixed.question.context.game_generation));
            owner
                .prepare_startup(deadline.min(Instant::now() + Duration::from_secs(30)))
                .map_err(|error| role_failure("cost_fresh_startup", error))?;
            let actual = handle.receipt();
            require(
                actual.startup_probe.as_ref().is_some_and(|v| {
                    v.completed_proposer_calls == 1
                        && v.completed_critic_calls == 1
                        && v.reset_completed
                        && v.runtime_mapping_confirmed
                        && v.cuda_placement_witness.is_some()
                }) && actual
                    .private_warm_observation
                    .as_ref()
                    .is_some_and(|v| v.accepted_seeds == 0)
                    && actual.search_consumed_role_inputs == 0,
                "cost_fresh_startup",
                "actual_cuda_readiness_or_missing_seed_not_verified",
            )?;
            startup = Some(actual);
            Ok::<_, Failure>(run_cost_lane(
                &mut owner, &handle, &trace, fixed, false, cancel,
            ))
        })();
        let work_completed_within_deadline =
            Instant::now() <= deadline && !cancel.load(Ordering::Acquire);
        let cleanup_started = Instant::now();
        let shutdown = handle.finish(whole);
        let receipt = handle.receipt();
        let observation = handle.cuda_warm_followup_observation();
        let joined = shutdown.is_ok()
            && receipt.physical_shutdown_confirmed
            && receipt.native_buffers_released
            && !receipt.quarantined
            && receipt.physical_runs_in_flight == 0;
        let mut lane =
            outcome.unwrap_or_else(|error| json!({"complete": false, "failure": error.wire()}));
        lane["source_identity"] = json!(source);
        lane["startup_receipt"] = json!(startup);
        lane["final_receipt"] = json!(receipt);
        lane["final_owner_observation"] = json!(observation);
        lane["physical_owner_joined"] = json!(joined);
        lane["work_completed_within_original_deadline"] = json!(work_completed_within_deadline);
        lane["cleanup_elapsed_ns"] = json!(cleanup_started.elapsed().as_nanos());
        lane["cleanup_failure"] = json!(
            shutdown
                .err()
                .map(|error| role_failure("cost_fresh_cleanup", error).wire())
        );
        lane["owner_total_actual_completed_nn_inputs"] = json!(
            receipt
                .backend_stats
                .as_ref()
                .map(|v| v.completed_nn_inputs)
        );
        lane["owner_total_backend_stats"] = json!(receipt.backend_stats);
        lane["complete"] =
            json!(lane["complete"] == true && joined && work_completed_within_deadline);
        drop(owner);
        drop(handle);
        Ok(lane)
    }
    fn execute_request(
        args: &Args,
        started: Instant,
        deadline: Instant,
        whole: Instant,
        cancel: &AtomicBool,
        capture: &Arc<Mutex<Capture>>,
    ) -> Result<Value, Failure> {
        let mut report = execute(args, started, deadline, whole, cancel, capture)?;
        if !args.cost_same_seed_context {
            return Ok(report);
        }
        let captured = capture
            .lock()
            .map_err(|_| fail("cost", "capture_unavailable"))?;
        let fixed = captured.cost_fixed_repair.clone();
        let mut warm = captured.cost_warm_lane.clone().unwrap_or_else(|| json!({
            "complete": false, "failure": fail("cost_source", "actual_accepted_repair_missing").wire(),
        }));
        drop(captured);
        let warm_joined = report["final_receipt"]["physical_shutdown_confirmed"] == true
            && report["final_receipt"]["native_buffers_released"] == true
            && report["final_receipt"]["physical_runs_in_flight"] == 0
            && report["final_receipt"]["quarantined"] == false
            && report["cuda_warm_owner_observation"]["backend_dropped"] == true
            && report["cuda_warm_owner_observation"]["worker_joined"] == true;
        warm["physical_owner_joined"] = json!(warm_joined);
        warm["final_receipt"] = report["final_receipt"].clone();
        warm["final_owner_observation"] = report["cuda_warm_owner_observation"].clone();
        warm["owner_total_backend_stats"] = report["final_receipt"]["backend_stats"].clone();
        warm["owner_total_actual_completed_nn_inputs"] =
            report["final_receipt"]["backend_stats"]["completed_nn_inputs"].clone();
        let fresh = match &fixed {
            Some(fixed) if warm_joined && warm["complete"] == true => {
                run_fresh_cost_owner(args, fixed, &warm, started, deadline, whole, cancel)
                    .unwrap_or_else(|error| json!({"complete": false, "failure": error.wire()}))
            }
            _ => {
                json!({"complete": false, "failure": fail("cost_fresh_load", "prior_warm_lane_or_physical_fence_incomplete").wire()})
            }
        };
        let complete = warm["complete"] == true
            && warm_joined
            && fresh["complete"] == true
            && Instant::now() <= whole
            && !cancel.load(Ordering::Acquire);
        let mut cost = json!({
            "opt_in": true, "scope": "repair_with_context_api_known_physical_logical_consumer_zero",
            "cost_profile": if args.cost_conv_default {"same-seed-conv-default-env0/v1"} else {"same-seed-legacy/v1"},
            "cudnn_conv_search_requested": if args.cost_conv_default {"default-device-warm-v1"} else {"legacy_implicit_exhaustive"},
            "tf32_requested": false, "tf32_actual": "unknown", "tf32_retry": "unknown",
            "nvidia_tf32_override_required": if args.cost_conv_default {Some("0")} else {None},
            "cudnn_execution_diagnostic": report["startup_receipt"]["execution"]["cudnn_conv_search_diagnostic"],
            "cost_execution_boundary": "resident_repair_actual_search_progress_before_later_P_publication",
            "execution_checks_passed": complete, "benchmark_validation_complete": false,
            "wall_limit_ms": cost_work_wall(args).as_millis(), "cleanup_limit_ms": CLEANUP.as_millis(),
            "whole_limit_ms": (cost_work_wall(args) + CLEANUP).as_millis(),
            "preparation_calls_per_lane": COST_PREPARATION_CALLS,
            "measured_calls_per_lane": COST_MEASURED_CALLS,
            "extra_role_forward_reservation_including_fresh_startup": COST_EXTRA_ROLE_FORWARDS,
            "owners_run_sequentially_after_actual_worker_join": warm_joined,
            "max_concurrent_physical_workers": 1,
            "selected_max_leases": 1, "selected_device_payload_bytes_max": COST_DEVICE_BYTES_MAX,
        });
        let reference = json!({
            "same_target_input_json_sha256_hex": fixed.as_ref().and_then(|v| sha_json(&v.input).ok()),
            "original_accepted_repair_source": fixed.as_ref().map(|v| &v.source),
            "original_accepted_full_6144_seed_bits": fixed.as_ref().map(|v| &v.source_latent),
            "same_seed_independent_reference_required": true,
            "independent_reference": "not_run;original_capture_same_seed_torch_and_ort_reference_preserved",
            "value_fresh_used_as_control": false,
            "fresh_control": "independent_warm_capable_factory_owner_with_no_accepted_seed",
            "warm_seed_preparation": "actual_accepted_repair_source_once_then_provisional_cost_outputs_discarded",
            "fresh_vs_warm_raw_output_equality_required": false,
            "observed_original_integration_calls": report["calls"].as_array().map(Vec::len),
            "warm_owner_role_totals_include_cost_calls": true,
            "fresh_owner_role_totals_reported_separately": true,
        });
        let comparison = json!({
            "sharing_measurement": null, "whole_device_vram_peak_bytes": null,
            "preparation_cache_counts_comparable": false, "comparable": false,
            "comparison_status": "not_comparable",
            "comparison_reason": "sample_phase_public_cache_and_nn_counts_not_exposed_by_live_native_receipt",
            "warm_over_fresh_api_wall_ratio": null,
            "api_wall_includes_native_prepare_submit_wait_decode_and_observer_cost": true,
            "physical_callback_timestamp_is_observer_receipt_time_not_cuda_kernel_duration": true,
        });
        // Keep macro expansion bounded; all three literals are known objects.
        if let (Value::Object(fields), Value::Object(reference), Value::Object(comparison)) =
            (&mut cost, reference, comparison)
        {
            fields.extend(reference);
            fields.extend(comparison);
        }
        cost["warm"] = warm;
        cost["fresh"] = fresh;
        report["cost_same_seed_context"] = cost;
        Ok(report)
    }

    fn failure_report(error: Failure, started: Instant) -> Value {
        json!({"schema": SCHEMA, "phase": "capture", "passed": false, "execution_checks_passed": false,
            "gpu_execution": "not_confirmed", "failure": error.wire(), "elapsed_ms": started.elapsed().as_millis(),
            "actual_training": false, "strength_or_speedup_claim": false})
    }
    fn execute(
        args: &Args,
        started: Instant,
        deadline: Instant,
        whole: Instant,
        cancel: &AtomicBool,
        capture: &Arc<Mutex<Capture>>,
    ) -> Result<Value, Failure> {
        // The dedicated DEFAULT cost namespace admits only the process-level
        // requested override. No ORT/cuDNN plan or actual TF32 state is inferred.
        cost_default_environment(args, std::env::var("NVIDIA_TF32_OVERRIDE").ok().as_deref())?;
        require(
            cfg!(target_os = "linux"),
            "platform",
            "linux_cuda_runtime_required",
        )?;
        let scenario_bytes = pinned(
            &args.path("--scenario")?,
            args.value("--scenario-sha256")?,
            64 * 1024,
        )?;
        let scenario: Scenario = serde_json::from_slice(&scenario_bytes)
            .map_err(|_| fail("scenario", "json_invalid"))?;
        let position = scenario.position()?;
        let owner_limits = args.cuda_warm_limits()?;
        let integration_forward_limit = if args.cost_same_seed_context {
            require(
                scenario.max_role_forwards == DEFAULT_ROLE_FORWARDS
                    && scenario.max_role_forwards
                        > COST_EXTRA_ROLE_FORWARDS + STARTUP_ROLE_FORWARDS + FOLLOWUP_ROLE_FORWARDS,
                "scenario",
                "cost_forward_reservation_out_of_bounds",
            )?;
            scenario.max_role_forwards - COST_EXTRA_ROLE_FORWARDS
        } else {
            scenario.max_role_forwards
        };
        capture
            .lock()
            .map_err(|_| fail("scenario", "capture_unavailable"))?
            .max_role_forwards = scenario.max_role_forwards;
        let bundle_bytes = pinned(
            &args.path("--cuda-bundle")?,
            args.value("--cuda-bundle-sha256")?,
            64 * 1024,
        )?;
        let spec = CudaRuntimeBundleSpec::from_json(
            std::str::from_utf8(&bundle_bytes)
                .map_err(|_| fail("runtime", "bundle_utf8_invalid"))?,
        )
        .map_err(|_| fail("runtime", "bundle_invalid"))?;
        let runtime_cache = RuntimeCache::open(&args.path("--runtime-output-root")?)
            .map_err(|_| fail("runtime", "runtime_cache_unavailable"))?;
        let pin = runtime_cache
            .cuda_bundle(&args.path("--runtime-root")?, &spec)
            .map_err(|_| fail("runtime", "bundle_pin_failed"))?;
        require(Instant::now() < deadline, "runtime", "deadline_after_pin")?;
        let runtime =
            OrtRuntime::load(&pin).map_err(|_| fail("runtime", "full_cuda_bootstrap_failed"))?;
        let control = if args.values.contains_key("--control-inventory") {
            Some(
                PalsCudaControlPolicy::from_inventory(
                    &args.path("--control-inventory")?,
                    args.value("--control-inventory-sha256")?,
                )
                .map_err(|_| fail("runtime", "control_inventory_invalid"))?,
            )
        } else {
            None
        };
        let profile_root = args.values.get("--profile-root").map(PathBuf::from);
        let load_started = Instant::now();
        let native =
            NativeRoleModel::load_with_runtime_and_cuda_private_warm_and_observations_diagnostic(
                &args.path("--export-manifest")?,
                args.value("--export-manifest-sha256")?,
                runtime,
                PalsOnnxConfig {
                    provider: Provider::Cuda {
                        device_id: 0,
                        arena_bytes: 256 * 1024 * 1024,
                    },
                    intra_threads: 1,
                    cache_public_memory: true,
                    device_public_memory: true,
                    cudnn_conv_search: cuda_conv_selection(args)?,
                },
                NativeOwnerOptions {
                    drain_limit: CLEANUP,
                    host_record_pages: None,
                },
                control.zip(profile_root.as_deref()),
                Some(owner_limits),
            )
            .map_err(cuda_warm_load_failure)?;
        let handle = native.finish_handle();
        let source = native.source_identity();
        let native = Arc::new(Mutex::new(native));
        let outcome = (|| {
            let mut selected = native
                .lock()
                .map_err(|_| fail("model", "owner_unavailable"))?;
            require(
                selected.model_config().profile == PalsModelProfile::FullLineInteractionV2
                    && selected.model_config().latent_elements() == LATENT_ELEMENTS,
                "model",
                "full_line_interaction_v2_required",
            )?;
            capture
                .lock()
                .map_err(|_| fail("model", "capture_unavailable"))?
                .configuration = selected.model_config().clone();
            selected
                .set_private_warm_observer(Box::new(Observer(Arc::clone(capture))))
                .map_err(|error| role_failure("observer", error))?;
            selected
                .configure_startup_probe_timeout(Some(30_000))
                .map_err(|error| role_failure("startup", error))?;
            selected
                .observe_startup_loading(load_started.elapsed())
                .map_err(|error| role_failure("startup", error))?;
            let identity = selected.identity().to_owned();
            let value_identity = selected
                .value_identity()
                .cloned()
                .ok_or(fail("model", "value_namespace_missing"))?;
            drop(selected);
            let forwards = Arc::new(ForwardBudget::new(integration_forward_limit));
            let model = Model {
                native: Arc::clone(&native),
                identity,
                value_identity,
                forwards: Arc::clone(&forwards),
            };
            let cpu_config = CpuConfig {
                profile: CpuProfile::Independent,
                tt_entries: 1024,
                max_depth: scenario.cpu_depth,
                quiescence_ply: 8,
            };
            let cpu = match scenario.execution_profile {
                ExecutionProfile::ActualOpponentContinuationV1 => {
                    CpuEngine::with_resume_policy(cpu_config, CpuResumePolicy::PausedStack)
                }
                ExecutionProfile::IterativeFrozenModelWdlV2 => CpuEngine::with_paused_side_scope(
                    cpu_config,
                    Arc::new(BootstrapCpuValue::default()),
                    CpuOrderingPolicy::LegacyMvvLvaV1,
                ),
            }
            .map_err(|_| fail("search", "own_cpu_configuration_failed"))?;
            let cpu = ObservedCpu {
                engine: cpu,
                capture: Arc::clone(capture),
            };
            let config = PalsConfig {
                beam_width: 1,
                line_plies: scenario.line_plies,
                max_nodes: 1024,
                max_records: 128,
                max_role_calls: integration_forward_limit
                    - STARTUP_ROLE_FORWARDS
                    - FOLLOWUP_ROLE_FORWARDS,
                cpu_nodes_per_task: scenario.cpu_nodes_per_task,
            };
            let mut engine = match scenario.execution_profile {
                ExecutionProfile::ActualOpponentContinuationV1 => {
                    PalsEngine::new_with_cpu_and_refinement_policy(
                        config,
                        model,
                        cpu,
                        PostRepairRecheckPolicy::ActualOpponentContinuationV1,
                    )
                }
                ExecutionProfile::IterativeFrozenModelWdlV2 => {
                    PalsEngine::new_with_boxed_checker_and_policies_and_cpu_interleave(
                        config,
                        model,
                        Box::new(
                            OwnedCpuChecker::new(cpu)
                                .map_err(|_| fail("search", "own_checker_configuration_failed"))?,
                        ),
                        ResolverPolicy::ModelWdlRestricted,
                        PostRepairRecheckPolicy::IterativeFrozenModelWdlV2,
                        CpuInterleavePolicy::ResumeAfterAcceptedRepairC,
                    )
                }
            }
            .map_err(|_| fail("search", "engine_configuration_failed"))?;
            engine
                .try_new_game(deadline, cancel)
                .map_err(|_| fail("search", "logical_game_reset_failed"))?;
            native
                .lock()
                .map_err(|_| fail("startup", "owner_unavailable"))?
                .prepare_startup(deadline.min(Instant::now() + Duration::from_secs(30)))
                .map_err(|error| role_failure("startup", error))?;
            let startup = handle.receipt();
            require(
                startup.execution.provider == "cuda"
                    && startup.execution.device_public_memory
                    && startup.execution.runtime_bundle_sha256.is_some()
                    && startup
                        .execution
                        .private_warm
                        .as_ref()
                        .is_some_and(|v| v.schema == PRIVATE_CUDA_WARM_SCHEMA)
                    && startup.startup_probe.as_ref().is_some_and(|v| {
                        v.completed_proposer_calls == 1
                            && v.completed_critic_calls == 1
                            && v.reset_completed
                            && v.runtime_mapping_confirmed
                            && v.cuda_placement_witness.is_some()
                    }),
                "startup",
                "actual_cuda_io_binding_readiness_missing",
            )?;
            capture
                .lock()
                .map_err(|_| fail("search", "capture_unavailable"))?
                .stage = "actual_pals_search";
            let mut progress_resume = None;
            let search = engine.search_with_observed_progress(
                &position,
                PalsLimits {
                    deadline,
                    max_rounds: scenario.max_rounds,
                    max_cpu_nodes: scenario.max_cpu_nodes,
                    cpu_depth: scenario.cpu_depth,
                },
                cancel,
                |_, stores, records| {
                    if progress_resume.is_some() {
                        return;
                    }
                    let selected = (|| {
                        let mut captured = capture
                            .lock()
                            .map_err(|_| fail("resume", "capture_unavailable"))?;
                        observe_scheduler_resume_links(&mut captured, stores)?;
                        let Some(resident) = resident_repair_request(
                            captured.calls.iter().map(|call| &call.wire),
                        ) else {
                            return Ok::<_, Failure>(None);
                        };
                        let proof = captured.cpu_resume_evidence_for_repair(Some(resident));
                        let selected = resident_repair_for_progress(
                            captured.calls.iter().map(|call| &call.wire),
                            &proof,
                        );
                        if selected.is_some() {
                            // Keep the final owner receipt bound to the exact
                            // chain selected at this live progress boundary.
                            captured.progress_replay_source = selected;
                        }
                        Ok::<_, Failure>(selected)
                    })();
                    let result = match selected {
                        Ok(None) => return,
                        Err(error) => Err(error),
                        Ok(Some(source)) => (|| {
                            let mut owner = native
                                .lock()
                                .map_err(|_| fail("resume", "owner_unavailable"))?;
                            let result = replay(
                                &mut owner,
                                &handle,
                                capture,
                                &forwards,
                                RepairProgress {
                                    records,
                                    stores,
                                    source_request: source,
                                },
                                cancel,
                            )?;
                            // Cost lanes consume the same actual resident Repair
                            // before the next P publication can evict its seed.
                            if args.cost_same_seed_context {
                                match prepare_warm_cost(&mut owner, &handle, capture, started, cancel) {
                                    Ok((fixed, lane)) => {
                                        let complete = lane["complete"] == true;
                                        let mut captured = capture
                                            .lock()
                                            .map_err(|_| fail("cost", "capture_unavailable"))?;
                                        captured.cost_fixed_repair = Some(fixed);
                                        captured.cost_warm_lane = Some(lane);
                                        require(complete, "cost", "progress_cost_lane_incomplete")?;
                                    }
                                    Err(error) => {
                                        capture
                                            .lock()
                                            .map_err(|_| fail("cost", "capture_unavailable"))?
                                            .cost_warm_lane =
                                            Some(json!({"complete": false, "failure": error.clone().wire()}));
                                        return Err(error);
                                    }
                                }
                            }
                            Ok(result)
                        })(),
                    };
                    if let Ok(mut captured) = capture.lock() {
                        captured.stage = "actual_pals_search";
                    }
                    if result.is_err() {
                        // Preserve the first failed probe and stop subsequent
                        // logical dispatch. Physical ownership drains below.
                        cancel.store(true, Ordering::Release);
                    }
                    progress_resume = Some(result);
                },
            );
            {
                let mut trace = capture
                    .lock()
                    .map_err(|_| fail("cpu_trace", "capture_unavailable"))?;
                observe_scheduler_resume_links(&mut trace, engine.stores())?;
            }
            let counters = engine
                .last_search_counters()
                .ok_or(fail("search", "actual_counters_missing"))?;
            let search_failure = search
                .as_ref()
                .err()
                .map(|error| pals_failure("search", "actual_search_failed", error.clone()));
            let search_wire = json!({"identity": engine.search_identity(), "policy": format!("{:?}", engine.post_repair_recheck_policy()),
                "returned_ok": search.is_ok(), "error": search_failure.as_ref().map(|error| error.code),
                "error_diagnostic": search_failure.clone().map(Failure::wire),
                "best_move": search.as_ref().ok().and_then(|v| v.best_move).map(|v| v.to_string()),
                "completion": search.as_ref().ok().map(|v| format!("{:?}", v.completion)),
                "role_calls": counters.role_calls, "proposer_calls": counters.proposer_calls,
                "critic_calls": counters.critic_calls, "repair_calls": counters.repair_calls,
                "accepted_repair_outputs": counters.accepted_repair_outputs, "accepted_critic_outputs": counters.accepted_critic_outputs,
                "cpu_tasks_requested": counters.cpu_tasks_requested, "completed_cpu_tasks": counters.completed_cpu_tasks,
                "partial_cpu_iterations": counters.partial_cpu_iterations, "cpu_nodes": counters.cpu_nodes,
                "repair_queue_enqueued": counters.repair_queue_enqueued, "repair_queue_without_new_evidence": counters.repair_queue_without_new_evidence,
                "paused_stack_selected": true, "execution_profile": format!("{:?}", scenario.execution_profile),
                "value_calls": counters.value_calls, "accepted_value_outputs": counters.accepted_value_outputs,
                "cpu_work_observation_incomplete": counters.cpu_work_observation_incomplete,
                "harness_forward_admission_budget": forwards.observation(),
                "cpu_interleave_policy": format!("{:?}", engine.cpu_interleave_policy()),
                "cpu_completed_iteration_side_scope": engine.cpu_side_scope_snapshot().map(|side| json!({
                    "attempts": side.attempts, "work": cpu_work(side.work),
                    "fixed_reserved_bytes": side.fixed_reserved_bytes,
                    "fixed_reserved_bytes_are_observed_peak": false, "complete": side.complete})),
                "actual_interrupted_stack_resume_trace": "actual_CpuSearcher_wrapper_and_TaskTable_resumed_from"});
            // No invented Repair, widened bank or relabeled Fresh fallback.
            // Missing actual progress remains an explicit unmet exercise gate.
            let resume = progress_resume.unwrap_or_else(|| {
                Err(fail(
                    "resume",
                    "actual_resident_repair_c_original_cpu_resume_progress_not_exercised",
                ))
            });
            // Preserve work observations even when the independent CPU owner
            // cannot prove shutdown. Native GPU cleanup below still has its own
            // physical receipt and must neither hide nor certify CPU closure.
            let (checker_closed, checker_cleanup_failure) = match engine.shutdown_checker(whole) {
                Ok(checker) => (
                    checker.cleanup_complete,
                    (!checker.cleanup_complete)
                        .then(|| fail("cleanup", "own_checker_cleanup_incomplete")),
                ),
                Err(error) => (
                    false,
                    Some(pals_failure(
                        "cleanup",
                        "own_checker_shutdown_failed",
                        error,
                    )),
                ),
            };
            drop(engine);
            native
                .lock()
                .map_err(|_| fail("cleanup", "owner_unavailable"))?
                .finish_search(if search.is_ok() && resume.is_ok() && checker_closed {
                    RoleSearchClosure::Completed
                } else {
                    RoleSearchClosure::Failed
                });
            let forwarded = forwards.load(Ordering::Acquire);
            Ok::<_, Failure>((
                startup,
                search_wire,
                search_failure,
                resume,
                checker_closed,
                checker_cleanup_failure,
                forwarded,
            ))
        })();
        let work_completed_within_deadline = Instant::now() <= deadline;
        // Same absolute cleanup deadline. Cancellation never releases ownership
        // or certifies a physical fence; finish supplies the actual receipt.
        let cleanup_started = Instant::now();
        let shutdown = handle.finish(whole);
        let receipt = handle.receipt();
        let owner_observation = handle.cuda_warm_followup_observation();
        let capture = capture
            .lock()
            .map_err(|_| fail("capture", "capture_unavailable"))?;
        let (
            startup,
            search,
            search_failure,
            resume,
            checker_closed,
            checker_cleanup_failure,
            forwarded,
            work_failure,
        ) = match outcome {
            Ok((startup, search, search_failure, resume, checker, checker_failure, count)) => (
                Some(startup),
                Some(search),
                search_failure,
                Some(resume),
                checker,
                checker_failure,
                Some(count),
                None,
            ),
            Err(error) => (None, None, None, None, false, None, None, Some(error)),
        };
        let captures_complete = !capture.calls.is_empty()
            && capture.calls.iter().all(|call| {
                call.wire["physical"]["dispatched"] == true
                    && call.wire["physical"]["ready"] == true
                    && call.wire["physical"]["completed_ok"] == true
                    && call.wire["physical"]["completion_unknown"] == false
                    && call.raw.is_some()
                    && call.accepted
            });
        let first_p_fresh = capture
            .calls
            .iter()
            .find(|call| call.wire["input"]["role"] == "proposer")
            .is_some_and(|call| call.wire["warm_start"] == false);
        let first_c_fresh = capture
            .calls
            .iter()
            .find(|call| call.wire["input"]["role"] == "critic")
            .is_some_and(|call| call.wire["warm_start"] == false);
        let nn_counts = receipt.backend_stats.as_ref().map(|v| {
            (
                v.role_nn_runs_attempted,
                v.role_nn_runs_completed,
                v.role_nn_runs_failed_known,
            )
        });
        let cost_calls = capture
            .cost_warm_lane
            .as_ref()
            .and_then(|lane| lane["calls"].as_array())
            .map_or(0, |calls| calls.len() as u64);
        let integration_forwards = capture.calls.len() as u64 + STARTUP_ROLE_FORWARDS;
        let expected_forwards = integration_forwards + cost_calls;
        let startup_completed = receipt
            .startup_probe
            .as_ref()
            .map(|v| v.completed_proposer_calls + v.completed_critic_calls);
        let accounting = nn_counts == Some((expected_forwards, expected_forwards, 0))
            && startup_completed == Some(STARTUP_ROLE_FORWARDS)
            && receipt.physically_completed_role_calls == capture.calls.len() as u64 + cost_calls
            && receipt.completed_role_inputs == capture.calls.len() as u64 + cost_calls
            && forwarded == Some(integration_forwards)
            && receipt.delivered_role_inputs == capture.calls.len() as u64 + cost_calls
            && receipt.search_consumed_role_inputs == capture.calls.len() as u64;
        let resume_ok = resume.as_ref().is_some_and(|v| v.is_ok());
        let cpu_resume = capture.cpu_resume_evidence();
        let all_value_fresh = capture
            .calls
            .iter()
            .filter(|call| {
                call.question.as_ref().is_some_and(|question| {
                    question.context.purpose == RoleQueryPurpose::ValueFresh
                })
            })
            .all(|call| {
                call.wire["warm_start"] == false && call.wire["invocation"]["mode"] == "fresh"
            });
        let cost_seed_consumptions = capture
            .cost_warm_lane
            .as_ref()
            .and_then(|lane| lane["calls"].as_array())
            .map_or(0, |calls| {
                calls
                    .iter()
                    .filter(|call| call["warm_start"] == true)
                    .count() as u64
            });
        let actual_seed_consumptions = cost_seed_consumptions
            + capture
                .calls
                .iter()
                .filter(|call| call.wire["warm_start"] == true)
                .count() as u64;
        let actual_value_calls = capture
            .calls
            .iter()
            .filter(|call| {
                call.question.as_ref().is_some_and(|question| {
                    question.context.purpose == RoleQueryPurpose::ValueFresh
                })
            })
            .count() as u64;
        // Join Native's existing owner DTO to this actual call stream. Neither
        // a capability declaration nor logical cancellation supplies a fence.
        let owner_accounting = owner_observation.as_ref().is_some_and(|owner| {
            owner.measurement_contract == PRIVATE_CUDA_WARM_MEASUREMENT_CONTRACT
                && owner.backend_owner_id != 0
                && owner.bank_owner_id != 0
                && owner.capability_available == Some(true)
                && owner.max_leases == owner_limits.max_leases
                && owner.device_bytes_max == owner_limits.device_bytes_max
                && owner.leases_admitted == expected_forwards
                && owner.leases_physically_completed == expected_forwards
                && owner.leases_quarantined == 0
                && owner.leases_active == 0
                && owner.leases_peak == 1
                && owner.startup_leases_admitted == STARTUP_ROLE_FORWARDS
                && owner.startup_leases_physically_completed == STARTUP_ROLE_FORWARDS
                && owner.startup_leases_quarantined == 0
                && owner.device_bytes_current == 0
                && owner.device_bytes_retained == 0
                && owner.device_bytes_peak > 0
                && owner.device_bytes_peak <= owner_limits.device_bytes_max
                && actual_seed_consumptions > 0
                && owner.accepted_seed_consumptions == actual_seed_consumptions
                && owner.seed_context_checked_consumptions == actual_seed_consumptions
                && owner.rejected_seed_contexts == 0
                && actual_value_calls > 0
                && owner.fresh_value_evaluations == actual_value_calls
                && owner.value_always_fresh
                && owner.admission_closed
                && owner.bank_closed
                && owner.backend_dropped
                && owner.worker_joined
                && owner.buffers_released
                && owner.complete
        });
        let execution_passed = work_failure.is_none()
            && work_completed_within_deadline
            && search.as_ref().is_some_and(|v| v["returned_ok"] == true)
            && resume_ok
            && checker_closed
            && capture.recheck_completed
            && cpu_resume["passed"] == true
            && all_value_fresh
            && captures_complete
            && first_p_fresh
            && first_c_fresh
            && accounting
            && owner_accounting
            && nn_counts.is_some_and(|counts| counts.0 <= capture.max_role_forwards)
            && !capture.output_reservation_exhausted
            && shutdown.is_ok()
            && receipt.physical_shutdown_confirmed
            && receipt.native_buffers_released
            && !receipt.quarantined
            && receipt.physical_runs_in_flight == 0
            && Instant::now() <= whole;
        Ok(json!({
            "schema": SCHEMA, "phase": "capture", "passed": false, "execution_checks_passed": execution_passed,
            "numeric_reference": "not_run;_verify_reference_requires_independent_same_seed_cpu_torch_and_ort",
            "provider": "cuda", "gpu_execution": receipt.startup_probe.as_ref().is_some_and(|v| v.cuda_placement_witness.is_some())
                && nn_counts.is_some_and(|counts| counts.1 > 0),
            "device_selection": {"device_id": 0, "selected_gpu_count": 1, "physical_worker_count": 1},
            "actual_training": false, "strength_or_speedup_claim": false, "public_sharing_or_vram_measurement_claim": false,
            "wall_limit_ms": cost_work_wall(args).as_millis(), "cleanup_limit_ms": CLEANUP.as_millis(), "whole_limit_ms": (cost_work_wall(args) + CLEANUP).as_millis(),
            "output_limit_bytes": OUTPUT_LIMIT, "max_actual_role_forwards_including_startup": capture.max_role_forwards,
            "default_role_forward_budget": DEFAULT_ROLE_FORWARDS,
            "role_forward_budget_bounds": {"min": MIN_ROLE_FORWARDS, "max": MAX_ROLE_FORWARDS},
            "search_role_and_value_forward_budget": integration_forward_limit - STARTUP_ROLE_FORWARDS - FOLLOWUP_ROLE_FORWARDS,
            "call_output_bytes_reserved": capture.call_output_bytes_reserved,
            "call_output_reservation_exhausted": capture.output_reservation_exhausted,
            "actual_role_forwards_including_startup": nn_counts.map(|counts| counts.0),
            "actual_completed_role_forwards_including_startup": nn_counts.map(|counts| counts.1),
            "actual_known_completed_startup_role_forwards": startup_completed,
            "role_forward_accounting_checked": accounting,
            "cuda_warm_owner_accounting_checked": owner_accounting,
            "cuda_warm_owner_observation": owner_observation,
            "cuda_warm_selected_limits": {"max_leases": owner_limits.max_leases,
                "device_bytes_max": owner_limits.device_bytes_max,
                "device_bytes_scope": "explicit_cuda_kv_payload_bytes"},
            "actual_native_checked_role_consumer_calls": receipt.search_consumed_role_inputs,
            "native_checked_consumer_scope": "actual_search_calls_plus_separately_labeled_same_question_resume_and_value_checks",
            "public_graph_runs_accounted_separately_by_backend_stats": true, "elapsed_ms": started.elapsed().as_millis(),
            "work_completed_within_deadline": work_completed_within_deadline,
            "cleanup_elapsed_ms": cleanup_started.elapsed().as_millis(),
            "harness_source_sha256_hex": hex(&Sha256::digest(include_bytes!("pals_cuda_warm_check.rs"))),
            "source_identity": source, "model_configuration": capture.configuration,
            "compiled_component_pins": {
                "model_boundary_sha256_hex": hex(&rz_eval::pals_model::compiled_model_boundary_sha256()),
                "search_implementation_sha256_hex": hex(&rz_search::pals::engine::compiled_search_implementation_sha256()),
                "resolver_implementation_sha256_hex": hex(&rz_search::pals::engine::compiled_resolver_implementation_sha256()),
                "cuda_warm_owner_implementation_sha256_hex": hex(&rz_eval::pals_onnx::cuda_private_warm_implementation_digest()),
                "native_cuda_warm_adapter_sha256_hex": hex(&rz_uci::pals_native::pals_native_cuda_warm_adapter_source_digest())},
            "export_manifest_sha256_hex": args.value("--export-manifest-sha256")?,
            "scenario_sha256_hex": args.value("--scenario-sha256")?, "cuda_bundle_manifest_sha256_hex": args.value("--cuda-bundle-sha256")?,
            "runtime_storage": pin.storage(),
            "first_proposer_fresh_checked": first_p_fresh, "first_critic_fresh_checked": first_c_fresh,
            "all_observed_value_calls_fresh_checked": all_value_fresh,
            "fresh_vs_seeded_equality_required": false, "same_seed_independent_reference_required": true,
            "startup_receipt": startup, "actual_search": search, "actual_search_rechecks": capture.rechecks,
            "resume": resume.as_ref().and_then(|v| v.as_ref().ok()),
            "failure": work_failure
                .or_else(|| resume.as_ref().and_then(|v| v.as_ref().err())
                    .filter(|error| error.code != "actual_resident_repair_c_original_cpu_resume_progress_not_exercised")
                    .cloned())
                .or(search_failure)
                .or_else(|| resume.as_ref().and_then(|v| v.as_ref().err().cloned()))
                .or_else(|| checker_cleanup_failure.clone())
                .or_else(|| (cpu_resume["passed"] != true).then(|| fail("cpu_scheduler_resume", "actual_pause_repair_c_resume_not_exercised")))
                .or_else(|| (!owner_accounting).then(|| fail("cuda_warm_owner", "actual_owner_accounting_or_closure_missing")))
                .map(Failure::wire),
            "cleanup_failure": shutdown.err().map(|error| role_failure("cleanup", error).wire()),
            "checker_cleanup_complete": checker_closed,
            "checker_cleanup_failure": checker_cleanup_failure.map(Failure::wire),
            "calls": capture.wires(), "final_receipt": receipt,
            "validator_v_execution": "not_run;_native_role_model_has_no_validator_call",
            "unknown_cancel_deadline_quarantine_evidence": "separate_mock_cpu_unit_tests;_not_exercised_by_this_actual_gpu_success_case",
            "scheduler_interrupted_stack_resume_after_gpu_repair": cpu_resume,
            "gpu_acceptance_missing_actual_resume": cpu_resume["passed"] != true,
            "tolerance": numeric_contract()
        }))
    }

    fn compare_slice(actual: &[f32], expected: &[f32]) -> Result<f64, Failure> {
        compare_slice_with(actual, expected, ATOL, RTOL)
    }
    fn compare_slice_with(
        actual: &[f32],
        expected: &[f32],
        atol: f64,
        rtol: f64,
    ) -> Result<f64, Failure> {
        require(
            actual.len() == expected.len(),
            "reference",
            "output_shape_differs",
        )?;
        let mut maximum: f64 = 0.0;
        for (&actual, &expected) in actual.iter().zip(expected) {
            require(
                actual.is_finite() && expected.is_finite(),
                "reference",
                "nonfinite_output",
            )?;
            let error = (f64::from(actual) - f64::from(expected)).abs();
            require(
                error <= atol + rtol * f64::from(expected).abs(),
                "reference",
                "numeric_mismatch",
            )?;
            maximum = maximum.max(error);
        }
        Ok(maximum)
    }
    fn compare_output(actual: &PalsRawOutput, expected: &PalsRawOutput) -> Result<f64, Failure> {
        require(
            actual.private_latent.len() == LATENT_ELEMENTS
                && expected.private_latent.len() == LATENT_ELEMENTS,
            "reference",
            "full_latent_required",
        )?;
        let mut error = compare_slice(&actual.candidate_logits, &expected.candidate_logits)?;
        error = error.max(compare_slice(&actual.wdl_logits, &expected.wdl_logits)?);
        error = error.max(compare_slice_with(
            &actual.private_latent,
            &expected.private_latent,
            LATENT_ATOL,
            LATENT_RTOL,
        )?);
        for (a, b) in [
            (
                actual.divergence_logits.as_deref(),
                expected.divergence_logits.as_deref(),
            ),
            (
                actual.task_logits.as_ref().map(|v| v.as_slice()),
                expected.task_logits.as_ref().map(|v| v.as_slice()),
            ),
        ] {
            match (a, b) {
                (Some(a), Some(b)) => error = error.max(compare_slice(a, b)?),
                (None, None) => {}
                _ => return Err(fail("reference", "role_head_presence_differs")),
            }
        }
        Ok(error)
    }
    fn compare_decoded(
        actual: &PalsRawOutput,
        expected: &PalsRawOutput,
        input: &PalsModelInput,
        config: &PalsModelConfig,
    ) -> Result<Value, Failure> {
        let a = actual
            .decode(input, config)
            .map_err(|_| fail("reference", "actual_decoded_output_invalid"))?;
        let b = expected
            .decode(input, config)
            .map_err(|_| fail("reference", "reference_decoded_output_invalid"))?;
        let policy_error = compare_slice_with(
            &a.candidate_policy,
            &b.candidate_policy,
            POLICY_WDL_ATOL,
            0.,
        )?;
        let wdl_error = compare_slice_with(&a.wdl, &b.wdl, POLICY_WDL_ATOL, 0.)?;
        Ok(json!({"legal_policy_max_abs_error": policy_error, "wdl_max_abs_error": wdl_error}))
    }
    fn cost_receipt_matches(
        receipt: &Value,
        calls: u64,
        consumers: u64,
        final_receipt: bool,
    ) -> bool {
        let live = receipt["quarantined"] == false
            && receipt["physical_runs_in_flight"].as_u64() == Some(0)
            && receipt["process_epoch"].as_u64().is_some()
            && receipt["game_generation"].as_u64().is_some()
            && [
                "physically_completed_role_calls",
                "completed_role_inputs",
                "delivered_role_inputs",
            ]
            .iter()
            .all(|key| receipt[key].as_u64() == Some(calls))
            && receipt["search_consumed_role_inputs"].as_u64() == Some(consumers);
        live && (!final_receipt
            || (receipt["physical_shutdown_confirmed"] == true
                && receipt["native_buffers_released"] == true
                && receipt["backend_stats"]["role_nn_runs_attempted"].as_u64()
                    == Some(calls + STARTUP_ROLE_FORWARDS)
                && receipt["backend_stats"]["role_nn_runs_completed"].as_u64()
                    == Some(calls + STARTUP_ROLE_FORWARDS)
                && receipt["backend_stats"]["role_nn_runs_failed_known"].as_u64() == Some(0)))
    }
    fn cost_raw_bits(raw: &PalsRawOutput) -> Value {
        json!({"candidate_logits": raw.candidate_logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "wdl_logits": raw.wdl_logits.map(f32::to_bits),
            "divergence_logits": raw.divergence_logits.as_ref().map(|v| v.iter().map(|v| v.to_bits()).collect::<Vec<_>>()),
            "task_logits": raw.task_logits.map(|v| v.map(f32::to_bits)),
            "private_latent": raw.private_latent.iter().map(|v| v.to_bits()).collect::<Vec<_>>()})
    }
    fn cost_profile_matches_reference(cost: &Value, reference: &Value) -> bool {
        match cost.get("cost_profile").and_then(Value::as_str) {
            None | Some("same-seed-legacy/v1") => matches!(
                reference.get("cost_profile").and_then(Value::as_str),
                None | Some("same-seed-legacy/v1")
            ),
            Some("same-seed-conv-default-env0/v1") => {
                reference["cost_profile"] == cost["cost_profile"]
                    && cost["cudnn_conv_search_requested"] == "default-device-warm-v1"
                    && cost["nvidia_tf32_override_required"] == "0"
                    && cost["tf32_requested"] == false
                    && cost["tf32_actual"] == "unknown"
                    && cost["tf32_retry"] == "unknown"
            }
            _ => false,
        }
    }
    fn cost_declared_bound_matches_capture(capture: &Value) -> bool {
        let cost = &capture["cost_same_seed_context"];
        let wall = match cost.get("cost_profile").and_then(Value::as_str) {
            None | Some("same-seed-legacy/v1") => COST_WALL,
            Some("same-seed-conv-default-env0/v1") => COST_CONV_DEFAULT_WALL,
            _ => return false,
        };
        cost["wall_limit_ms"].as_u64() == u64::try_from(wall.as_millis()).ok()
            && capture["wall_limit_ms"] == cost["wall_limit_ms"]
            && cost["cleanup_limit_ms"].as_u64() == u64::try_from(CLEANUP.as_millis()).ok()
            && capture["cleanup_limit_ms"] == cost["cleanup_limit_ms"]
            && cost["whole_limit_ms"].as_u64() == u64::try_from((wall + CLEANUP).as_millis()).ok()
            && capture["whole_limit_ms"] == cost["whole_limit_ms"]
    }
    fn cost_sample_origin(
        cost: &Value,
        lane: &Value,
        warm: bool,
        final_integration_inputs: u64,
    ) -> Result<u64, Failure> {
        if !warm {
            return Ok(0);
        }
        if cost.get("cost_execution_boundary").is_none() {
            // Legacy artifacts ran cost after all integration inputs. Keep
            // their exact counter interpretation, without rewriting them.
            return Ok(final_integration_inputs);
        }
        let base = lane["integration_inputs_at_progress"]
            .as_u64()
            .ok_or(fail("cost_reference", "progress_cost_origin_missing"))?;
        require(
            cost["cost_execution_boundary"]
                == "resident_repair_actual_search_progress_before_later_P_publication"
                && base > 0
                && base <= final_integration_inputs
                && cost_receipt_matches(&lane["receipt_at_resident_progress"], base, base, false),
            "cost_reference",
            "progress_cost_origin_or_known_consumer_count_invalid",
        )?;
        Ok(base)
    }
    fn cost_reference(
        capture: &Value,
        reference: &Value,
        configuration: &PalsModelConfig,
    ) -> Result<Value, Failure> {
        let cost = &capture["cost_same_seed_context"];
        let independent = &reference["cost_same_seed_context"];
        let same_harness_source = capture["harness_source_sha256_hex"]
            == hex(&Sha256::digest(include_bytes!("pals_cuda_warm_check.rs")));
        // Historical records retain their original numerical verification path.
        // They are not admissions under this source's new DEFAULT time bound.
        require(
            !same_harness_source || cost_declared_bound_matches_capture(capture),
            "cost_reference",
            "same_harness_declared_time_bound_differs",
        )?;
        let originals = capture["calls"]
            .as_array()
            .ok_or(fail("cost_reference", "original_calls_missing"))?;
        let source = &cost["original_accepted_repair_source"];
        let latent: Vec<u32> =
            serde_json::from_value(cost["original_accepted_full_6144_seed_bits"].clone())
                .map_err(|_| fail("cost_reference", "original_seed_missing"))?;
        let original_raw: PalsRawOutput = serde_json::from_value(source["raw_output"].clone())
            .map_err(|_| fail("cost_reference", "original_output_missing"))?;
        require(
            cost["opt_in"] == true
                && cost["execution_checks_passed"] == true
                && cost["owners_run_sequentially_after_actual_worker_join"] == true
                && cost["max_concurrent_physical_workers"].as_u64() == Some(1)
                && cost["selected_max_leases"].as_u64() == Some(1)
                && cost["selected_device_payload_bytes_max"].as_u64()
                    == Some(COST_DEVICE_BYTES_MAX)
                && cost["preparation_calls_per_lane"].as_u64()
                    == Some(COST_PREPARATION_CALLS as u64)
                && cost["measured_calls_per_lane"].as_u64() == Some(COST_MEASURED_CALLS as u64)
                && cost["extra_role_forward_reservation_including_fresh_startup"].as_u64()
                    == Some(COST_EXTRA_ROLE_FORWARDS)
                && capture["max_actual_role_forwards_including_startup"].as_u64()
                    == Some(DEFAULT_ROLE_FORWARDS)
                && originals.len() as u64 + STARTUP_ROLE_FORWARDS
                    <= DEFAULT_ROLE_FORWARDS - COST_EXTRA_ROLE_FORWARDS
                && originals.contains(source)
                && source["accepted"] == true
                && source["accepted_with_original_controls"] == true
                && source["delivered"] == true
                && source["physical"]["dispatched"] == true
                && source["physical"]["ready"] == true
                && source["physical"]["completed_ok"] == true
                && source["physical"]["completion_unknown"] == false
                && source["logical_context"]["purpose"] == "RepairPolicy"
                && latent.len() == LATENT_ELEMENTS
                && latent.iter().all(|v| f32::from_bits(*v).is_finite())
                && original_raw
                    .private_latent
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
                    == latent
                && source["raw_output_bits"] == cost_raw_bits(&original_raw)
                && cost["same_target_input_json_sha256_hex"] == source["input_json_sha256_hex"]
                && independent["opt_in"] == true
                && independent["independent_same_seed_reference_passed"] == true
                && independent["original_accepted_repair_source"] == *source
                && independent["original_accepted_full_6144_seed_bits"]
                    == cost["original_accepted_full_6144_seed_bits"]
                && independent["same_target_input_json_sha256_hex"]
                    == cost["same_target_input_json_sha256_hex"]
                && cost_profile_matches_reference(cost, independent),
            "cost_reference",
            "partial_or_unbound_original_cost_source",
        )?;
        let original_count = originals.len() as u64;
        let expected_warm = original_count + COST_LANE_CALLS as u64;
        require(
            capture["actual_role_forwards_including_startup"].as_u64()
                == Some(expected_warm + STARTUP_ROLE_FORWARDS)
                && capture["actual_completed_role_forwards_including_startup"].as_u64()
                    == Some(expected_warm + STARTUP_ROLE_FORWARDS)
                && capture["actual_known_completed_startup_role_forwards"].as_u64()
                    == Some(STARTUP_ROLE_FORWARDS)
                && capture["role_forward_accounting_checked"] == true,
            "cost_reference",
            "warm_owner_and_original_integration_accounting_differ",
        )?;
        let expected_input: PalsModelInput = serde_json::from_value(source["input"].clone())
            .map_err(|_| fail("cost_reference", "source_input_invalid"))?;
        let mut result = json!({"opt_in": true, "independent_same_seed_reference_passed": true,
            "comparison_status": "not_comparable", "warm_over_fresh_api_wall_ratio": null,
            "physical_nn_graph_inputs_per_sample": null, "public_cache_counts_per_sample": null,
            "benchmark_validation_complete": false, "fresh_vs_warm_output_equality_required": false});
        result["declared_time_bound_checked"] = json!(same_harness_source);
        result["time_bound_check_scope"] = json!(if same_harness_source {
            "same_harness_declared_work_and_cleanup_bounds"
        } else {
            "historical_capture_numeric_only;_no_current_time_bound_admission"
        });
        for name in ["wall_limit_ms", "cleanup_limit_ms", "whole_limit_ms"] {
            result[name] = cost[name].clone();
        }
        if let Some(profile) = cost.get("cost_profile") {
            result["cost_profile"] = profile.clone();
            result["tf32_requested"] = cost["tf32_requested"].clone();
            result["tf32_actual"] = cost["tf32_actual"].clone();
            result["tf32_retry"] = cost["tf32_retry"].clone();
            result["cudnn_execution_diagnostic"] = cost["cudnn_execution_diagnostic"].clone();
        }
        for (name, warm) in [("warm", true), ("fresh", false)] {
            let lane = &cost[name];
            let lane_reference = &independent[name];
            let calls = lane["calls"]
                .as_array()
                .ok_or(fail("cost_reference", "lane_calls_missing"))?;
            let references = lane_reference["calls"]
                .as_array()
                .ok_or(fail("cost_reference", "lane_reference_missing"))?;
            let base = if warm { original_count } else { 0 };
            let consumers = base;
            let sample_base = cost_sample_origin(cost, lane, warm, original_count)?;
            require(
                lane["complete"] == true
                    && lane["failure"].is_null()
                    && lane["physical_owner_joined"] == true
                    && lane["preparation_calls_requested"].as_u64()
                        == Some(COST_PREPARATION_CALLS as u64)
                    && lane["measured_calls_requested"].as_u64()
                        == Some(COST_MEASURED_CALLS as u64)
                    && lane["measured_calls_completed"].as_u64()
                        == Some(COST_MEASURED_CALLS as u64)
                    && calls.len() == COST_LANE_CALLS
                    && references.len() == COST_LANE_CALLS
                    && cost_receipt_matches(
                        &lane["final_receipt"],
                        base + COST_LANE_CALLS as u64,
                        consumers,
                        true,
                    ),
                "cost_reference",
                "incomplete_lane_or_owner_accounting",
            )?;
            let source_identity = if warm {
                &capture["source_identity"]
            } else {
                &lane["source_identity"]
            };
            require(
                !source_identity.is_null()
                    && lane_reference["source_identity"] == *source_identity
                    && source_identity["checkpoint_sha256"]
                        == capture["source_identity"]["checkpoint_sha256"]
                    && source_identity["export_manifest_sha256"]
                        == capture["source_identity"]["export_manifest_sha256"]
                    && source_identity["model_configuration"] == capture["model_configuration"],
                "cost_reference",
                "lane_model_source_namespace_differs",
            )?;
            if warm {
                require(
                    lane["final_receipt"] == capture["final_receipt"],
                    "cost_reference",
                    "warm_final_owner_differs",
                )?;
            } else {
                let startup = &lane["startup_receipt"];
                require(
                    cost_receipt_matches(startup, 0, 0, false)
                        && startup["private_warm_observation"]["accepted_seeds"].as_u64()
                            == Some(0)
                        && startup["startup_probe"]["completed_proposer_calls"].as_u64() == Some(1)
                        && startup["startup_probe"]["completed_critic_calls"].as_u64() == Some(1)
                        && startup["startup_probe"]["reset_completed"] == true
                        && startup["startup_probe"]["runtime_mapping_confirmed"] == true
                        && !startup["startup_probe"]["cuda_placement_witness"].is_null()
                        && lane["work_completed_within_original_deadline"] == true
                        && lane["cleanup_failure"].is_null(),
                    "cost_reference",
                    "fresh_startup_or_original_cleanup_incomplete",
                )?;
            }
            let mut stable = None;
            // Warm cost uses the original physical owner; Fresh is independent.
            // Seed validation only, preserving every captured raw ID.
            let mut ids: Vec<(Value, Value)> = if warm {
                originals
                    .iter()
                    .map(|call| (call["request_id"].clone(), call["execution_id"].clone()))
                    .collect()
            } else {
                Vec::with_capacity(COST_LANE_CALLS)
            };
            let mut checks = Vec::with_capacity(COST_LANE_CALLS);
            for (index, (call, packet)) in calls.iter().zip(references).enumerate() {
                let sample = &call["cost_sample"];
                let before = &sample["receipt_before"];
                let returned = &sample["receipt_returned"];
                let after = &sample["receipt_after_close"];
                require(
                    sample["lane"]
                        == if warm {
                            "warm_same_accepted_seed"
                        } else {
                            "fresh_missing_seed"
                        }
                        && sample["phase"]
                            == if index < COST_PREPARATION_CALLS {
                                "preparation"
                            } else {
                                "measurement"
                            }
                        && sample["ordinal_in_phase"].as_u64()
                            == Some(if index < COST_PREPARATION_CALLS {
                                index + 1
                            } else {
                                index + 1 - COST_PREPARATION_CALLS
                            } as u64)
                        && sample["api_returned_ok"] == true
                        && sample["close_unconsumed_returned_ok"] == true
                        && cost_receipt_matches(
                            before,
                            sample_base + index as u64,
                            sample_base,
                            false,
                        )
                        && cost_receipt_matches(
                            returned,
                            sample_base + index as u64 + 1,
                            sample_base,
                            false,
                        )
                        && cost_receipt_matches(
                            after,
                            sample_base + index as u64 + 1,
                            sample_base,
                            false,
                        )
                        && ["process_epoch", "game_generation"].iter().all(|key| {
                            before[key] == returned[key]
                                && before[key] == after[key]
                                && before[key] == lane["final_receipt"][key]
                        })
                        && before["game_generation"].as_u64().is_some()
                        && before["game_generation"]
                            == source["logical_context"]["game_generation"]
                        && cost_known_unconsumed(before, after, call)
                        && fixed_cost_packet(
                            call,
                            &source["input"],
                            &source["logical_context"],
                            &latent,
                            source,
                            warm,
                            &mut stable,
                        )
                        && (!warm
                            || after["private_warm_observation"]["accepted_seeds"]
                                .as_u64()
                                .is_some_and(|v| v > 0))
                        && (warm
                            || after["private_warm_observation"]["accepted_seeds"].as_u64()
                                == Some(0)),
                    "cost_reference",
                    "sample_input_seed_completion_or_consumer_zero_differs",
                )?;
                require(
                    !ids.iter().any(|(request, execution)| {
                        request == &call["request_id"] || execution == &call["execution_id"]
                    }) && ["request_id", "execution_id"].iter().all(|key| {
                        call[key]["epoch"].as_u64().is_some()
                            && call[key]["sequence"].as_u64().is_some()
                    }),
                    "cost_reference",
                    "duplicate_or_unknown_lane_request_identity",
                )?;
                ids.push((call["request_id"].clone(), call["execution_id"].clone()));
                require(
                    [
                        "request_id",
                        "execution_id",
                        "input_key_hex",
                        "input_json_sha256_hex",
                        "input_f32_bits",
                        "invocation",
                        "initial_latent_sha256_hex",
                        "warm_start",
                        "input_json_utf8",
                        "initial_latent_bits",
                        "logical_context",
                        "seed_provenance",
                    ]
                    .iter()
                    .all(|key| call[key] == packet[key])
                        && packet["full_line"] == call["input"]["full_line"]
                        && packet["captured_raw_output_bits"] == call["raw_output_bits"]
                        && ["lane", "phase", "ordinal_in_phase"]
                            .iter()
                            .all(|key| sample[key] == packet["cost_sample"][key]),
                    "cost_reference",
                    "exact_lane_reference_packet_differs",
                )?;
                let input_text = call["input_json_utf8"]
                    .as_str()
                    .ok_or(fail("cost_reference", "input_bytes_missing"))?;
                let input: PalsModelInput = serde_json::from_str(input_text)
                    .map_err(|_| fail("cost_reference", "input_bytes_invalid"))?;
                require(
                    input == expected_input
                        && input_f32_bits(&input) == call["input_f32_bits"]
                        && hex(&Sha256::digest(input_text.as_bytes()))
                            == call["input_json_sha256_hex"]
                        && input
                            .canonical_input_key(configuration)
                            .ok()
                            .map(|key| hex(&key))
                            .as_deref()
                            == call["input_key_hex"].as_str(),
                    "cost_reference",
                    "input_utf8_bits_or_key_differs",
                )?;
                let checkpoint: [u8; 32] = serde_json::from_value(
                    source_identity["checkpoint_sha256"].clone(),
                )
                .map_err(|_| fail("cost_reference", "owner_checkpoint_namespace_missing"))?;
                let manifest: [u8; 32] =
                    serde_json::from_value(source_identity["export_manifest_sha256"].clone())
                        .map_err(|_| fail("cost_reference", "owner_manifest_namespace_missing"))?;
                require(
                    checkpoint == input.model_epoch
                        && hex(&manifest) == capture["export_manifest_sha256_hex"],
                    "cost_reference",
                    "lane_input_and_model_owner_namespace_differ",
                )?;
                if warm {
                    let encoding: [u8; 32] =
                        serde_json::from_value(source_identity["encoding_semantic_sha256"].clone())
                            .map_err(|_| {
                                fail("cost_reference", "owner_encoding_namespace_missing")
                            })?;
                    require(
                        call["seed_provenance"]["role"] == call["input"]["role"]
                            && call["seed_provenance"]["model_manifest_sha256_hex"]
                                == capture["export_manifest_sha256_hex"]
                            && call["seed_provenance"]["model_epoch_hex"]
                                == hex(&input.model_epoch)
                            && call["seed_provenance"]["encoding_semantic_sha256_hex"]
                                == hex(&encoding)
                            && call["seed_provenance"]["frozen_epoch"].as_u64().is_some()
                            && call["seed_provenance"]["frozen_epoch"]
                                == source_identity["frozen_epoch"],
                        "cost_reference",
                        "seed_role_model_owner_namespace_differs",
                    )?;
                }
                let actual: PalsRawOutput = serde_json::from_value(call["raw_output"].clone())
                    .map_err(|_| fail("cost_reference", "raw_output_invalid"))?;
                let ort: PalsRawOutput =
                    serde_json::from_value(packet["ort_raw_output"].clone())
                        .map_err(|_| fail("cost_reference", "ort_output_invalid"))?;
                let torch: PalsRawOutput =
                    serde_json::from_value(packet["torch_raw_output"].clone())
                        .map_err(|_| fail("cost_reference", "torch_output_invalid"))?;
                require(
                    cost_raw_bits(&actual) == call["raw_output_bits"]
                        && call["final_latent_sha256_hex"]
                            == bits_digest(
                                &actual
                                    .private_latent
                                    .iter()
                                    .map(|v| v.to_bits())
                                    .collect::<Vec<_>>(),
                            ),
                    "cost_reference",
                    "raw_output_full_bits_or_digest_differs",
                )?;
                let ort_error = compare_output(&actual, &ort)?;
                let torch_error = compare_output(&actual, &torch)?;
                compare_output(&ort, &torch)?;
                let ort_decoded = compare_decoded(&actual, &ort, &input, configuration)?;
                let torch_decoded = compare_decoded(&actual, &torch, &input, configuration)?;
                compare_decoded(&ort, &torch, &input, configuration)?;
                checks.push(json!({"request_id": call["request_id"], "execution_id": call["execution_id"],
                    "cost_sample": packet["cost_sample"], "input_key_hex": call["input_key_hex"],
                    "initial_latent_sha256_hex": call["initial_latent_sha256_hex"],
                    "cuda_vs_cpu_ort_max_abs_error": ort_error, "cuda_vs_cpu_torch_max_abs_error": torch_error,
                    "cuda_vs_cpu_ort_decoded": ort_decoded, "cuda_vs_cpu_torch_decoded": torch_decoded, "passed": true}));
            }
            require(
                !warm || lane["stable_accepted_seed_provenance"] == json!(stable),
                "cost_reference",
                "stable_provenance_differs",
            )?;
            result[name] = json!({"checks": checks, "source_identity": source_identity, "logical_search_consumers_per_sample": 0});
        }
        Ok(completed_cost_reference_metadata(result))
    }
    fn completed_cost_reference_metadata(mut result: Value) -> Value {
        // Called only after every original, owner, same-seed and numeric check.
        result["execution_checks_passed"] = json!(true);
        result
    }
    fn reference(args: &Args) -> Result<Value, Failure> {
        let capture_bytes = pinned(
            &args.path("--capture-json")?,
            args.value("--capture-sha256")?,
            OUTPUT_LIMIT,
        )?;
        let reference_bytes = pinned(
            &args.path("--independent-reference")?,
            args.value("--independent-reference-sha256")?,
            OUTPUT_LIMIT,
        )?;
        let capture: Value = serde_json::from_slice(&capture_bytes)
            .map_err(|_| fail("reference", "capture_json_invalid"))?;
        let reference: Value = serde_json::from_slice(&reference_bytes)
            .map_err(|_| fail("reference", "reference_json_invalid"))?;
        require(
            capture["schema"] == SCHEMA
                && capture["phase"] == "capture"
                && capture["execution_checks_passed"] == true
                && capture["cuda_warm_owner_accounting_checked"] == true
                && capture["scheduler_interrupted_stack_resume_after_gpu_repair"]["passed"] == true,
            "reference",
            "actual_execution_gate_failed",
        )?;
        require(
            reference["schema"] == REFERENCE_SCHEMA
                && reference["capture_sha256_hex"] == args.value("--capture-sha256")?
                && reference["export_manifest_sha256_hex"] == capture["export_manifest_sha256_hex"]
                && reference["model_configuration"] == capture["model_configuration"]
                && reference["max_actual_role_forwards_including_startup"]
                    == capture["max_actual_role_forwards_including_startup"]
                && reference["reference_provider"] == "cpu"
                && reference["reference_graph_execution"]
                    == "onnxruntime_and_torch_same_actual_inputs_and_seed",
            "reference",
            "independent_reference_identity_differs",
        )?;
        let calls = capture["calls"]
            .as_array()
            .ok_or(fail("reference", "capture_calls_missing"))?;
        let references = reference["calls"]
            .as_array()
            .ok_or(fail("reference", "reference_calls_missing"))?;
        let configuration: PalsModelConfig =
            serde_json::from_value(capture["model_configuration"].clone())
                .map_err(|_| fail("reference", "model_configuration_invalid"))?;
        require(
            configuration.profile == PalsModelProfile::FullLineInteractionV2,
            "reference",
            "full_line_interaction_v2_required",
        )?;
        let role_budget = capture["max_actual_role_forwards_including_startup"]
            .as_u64()
            .ok_or(fail("reference", "role_budget_missing"))?;
        require(
            (MIN_ROLE_FORWARDS..=MAX_ROLE_FORWARDS).contains(&role_budget),
            "reference",
            "role_budget_out_of_bounds",
        )?;
        require(
            !calls.is_empty()
                && calls.len() as u64 <= role_budget - STARTUP_ROLE_FORWARDS
                && calls.len() == references.len(),
            "reference",
            "call_count_differs",
        )?;
        let mut results = Vec::with_capacity(calls.len());
        for (call, reference) in calls.iter().zip(references) {
            require(
                call["request_id"] == reference["request_id"]
                    && call["execution_id"] == reference["execution_id"]
                    && call["input_key_hex"] == reference["input_key_hex"]
                    && call["input_json_sha256_hex"] == reference["input_json_sha256_hex"]
                    && call["invocation"] == reference["invocation"]
                    && call["input_f32_bits"] == reference["input_f32_bits"]
                    && call["input"]["full_line"] == reference["full_line"]
                    && call["initial_latent_sha256_hex"] == reference["initial_latent_sha256_hex"]
                    && call["warm_start"] == reference["warm_start"],
                "reference",
                "actual_request_input_or_seed_differs",
            )?;
            let input_text = call["input_json_utf8"]
                .as_str()
                .ok_or(fail("reference", "actual_input_bytes_missing"))?;
            let input: PalsModelInput = serde_json::from_str(input_text)
                .map_err(|_| fail("reference", "actual_input_json_invalid"))?;
            let structured: PalsModelInput = serde_json::from_value(call["input"].clone())
                .map_err(|_| fail("reference", "actual_structured_input_invalid"))?;
            require(
                input == structured
                    && input_f32_bits(&input) == call["input_f32_bits"]
                    && input_f32_bits(&structured) == call["input_f32_bits"]
                    && hex(&Sha256::digest(input_text.as_bytes())) == call["input_json_sha256_hex"]
                    && input
                        .canonical_input_key(&configuration)
                        .ok()
                        .map(|key| hex(&key))
                        .as_deref()
                        == call["input_key_hex"].as_str(),
                "reference",
                "exact_input_bytes_float_bits_tokens_or_key_differs",
            )?;
            let actual: PalsRawOutput = serde_json::from_value(call["raw_output"].clone())
                .map_err(|_| fail("reference", "actual_raw_output_invalid"))?;
            let ort: PalsRawOutput = serde_json::from_value(reference["ort_raw_output"].clone())
                .map_err(|_| fail("reference", "ort_raw_output_invalid"))?;
            let torch: PalsRawOutput =
                serde_json::from_value(reference["torch_raw_output"].clone())
                    .map_err(|_| fail("reference", "torch_raw_output_invalid"))?;
            let ort_error = compare_output(&actual, &ort)?;
            let torch_error = compare_output(&actual, &torch)?;
            compare_output(&ort, &torch)?;
            let ort_decoded = compare_decoded(&actual, &ort, &input, &configuration)?;
            let torch_decoded = compare_decoded(&actual, &torch, &input, &configuration)?;
            compare_decoded(&ort, &torch, &input, &configuration)?;
            results.push(json!({"request_id": call["request_id"], "warm_start": call["warm_start"],
                "input_key_hex": call["input_key_hex"], "initial_latent_sha256_hex": call["initial_latent_sha256_hex"],
                "input_json_sha256_hex": call["input_json_sha256_hex"], "full_line_tokens_and_float_bits_exactly_checked": true,
                "cuda_vs_cpu_ort_max_abs_error": ort_error, "cuda_vs_cpu_torch_max_abs_error": torch_error,
                "cuda_vs_cpu_ort_decoded": ort_decoded, "cuda_vs_cpu_torch_decoded": torch_decoded, "passed": true}));
        }
        let cost_checked = capture
            .get("cost_same_seed_context")
            .map(|_| cost_reference(&capture, &reference, &configuration))
            .transpose()?;
        require(
            cost_checked.is_some() || reference.get("cost_same_seed_context").is_none(),
            "cost_reference",
            "unexpected_cost_reference_namespace",
        )?;
        let mut report = json!({"schema": SCHEMA, "phase": "verify_reference", "passed": true,
            "execution_checks_passed": true, "independent_same_seed_reference_passed": true,
            "capture_sha256_hex": args.value("--capture-sha256")?,
            "independent_reference_sha256_hex": args.value("--independent-reference-sha256")?,
            "export_manifest_sha256_hex": capture["export_manifest_sha256_hex"], "model_configuration": capture["model_configuration"],
            "max_actual_role_forwards_including_startup": role_budget,
            "harness_source_sha256_hex": hex(&Sha256::digest(include_bytes!("pals_cuda_warm_check.rs"))),
            "tolerance": numeric_contract(), "checks": results,
            "fresh_vs_seeded_equality_required": false, "actual_training": false, "strength_or_speedup_claim": false,
            "scheduler_interrupted_stack_resume_after_gpu_repair": capture["scheduler_interrupted_stack_resume_after_gpu_repair"],
            "cuda_warm_owner_accounting_checked": capture["cuda_warm_owner_accounting_checked"],
            "cuda_warm_owner_observation": capture["cuda_warm_owner_observation"],
            "cuda_warm_selected_limits": capture["cuda_warm_selected_limits"],
            "validator_v_execution": capture["validator_v_execution"],
            "unknown_cancel_deadline_quarantine_evidence": capture["unknown_cancel_deadline_quarantine_evidence"]});
        if let Some(cost) = cost_checked {
            report["cost_same_seed_context"] = cost;
        }
        Ok(report)
    }
    fn report_passes_execution_gate(report: &Value) -> bool {
        (report["passed"] == true
            || (report["phase"] == "capture" && report["execution_checks_passed"] == true))
            && report
                .get("cost_same_seed_context")
                .is_none_or(|cost| cost["execution_checks_passed"] == true)
    }
    fn publish(report: Value, output_path: Option<&Path>) -> ExitCode {
        let bytes = match serde_json::to_vec(&report) {
            Ok(bytes) if bytes.len() < OUTPUT_LIMIT => bytes,
            _ => {
                eprintln!("bounded CUDA Warm report serialization failed");
                return ExitCode::FAILURE;
            }
        };
        if let Some(path) = output_path {
            let write = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)
                .and_then(|mut file| file.write_all(&bytes).and_then(|_| file.sync_all()));
            if write.is_err() {
                eprintln!("exclusive CUDA Warm report write failed");
                return ExitCode::FAILURE;
            }
        }
        // Full raw tensors live in the bounded artifact. Avoid a second 128 MiB
        // stdout copy; native logs remain subject to the parent's pipe guard.
        let mut summary = json!({"schema": SCHEMA, "phase": report["phase"], "passed": report["passed"],
            "execution_checks_passed": report["execution_checks_passed"], "failure": report["failure"],
            "report_sha256_hex": hex(&Sha256::digest(&bytes)), "report_bytes": bytes.len(),
            "output_json": output_path.map(|v| v.to_string_lossy().into_owned())});
        if let Some(cost) = report.get("cost_same_seed_context") {
            summary["cost_execution_checks_passed"] = cost["execution_checks_passed"].clone();
            summary["cost_comparison_status"] = cost["comparison_status"].clone();
        }
        let stdout = serde_json::to_vec(&summary).unwrap_or_default();
        let mut stdout_writer = io::stdout().lock();
        if stdout_writer
            .write_all(&stdout)
            .and_then(|_| stdout_writer.write_all(b"\n"))
            .is_err()
        {
            return ExitCode::FAILURE;
        }
        if report_passes_execution_gate(&report) {
            ExitCode::SUCCESS
        } else {
            ExitCode::FAILURE
        }
    }
    pub fn main() -> ExitCode {
        let started = Instant::now();
        let args = match args() {
            Ok(args) => args,
            Err(error) => return publish(failure_report(error, started), None),
        };
        if args.mode == "verify-reference" {
            let report = reference(&args).unwrap_or_else(|error| {
                json!({"schema": SCHEMA, "phase": "verify_reference",
                "passed": false, "execution_checks_passed": false, "failure": error.wire()})
            });
            return publish(report, Some(&args.output));
        }
        let deadline = started + cost_work_wall(&args);
        let whole = deadline + CLEANUP;
        let cancel = Arc::new(AtomicBool::new(false));
        let capture = Arc::new(Mutex::new(Capture::new(started)));
        if args.cost_same_seed_context {
            if let Ok(mut capture) = capture.lock() {
                capture.cost_same_seed_context = true;
            }
        }
        let output = args.output.clone();
        let cost_selected = args.cost_same_seed_context;
        let worker_cancel = Arc::clone(&cancel);
        let worker_capture = Arc::clone(&capture);
        let (send, receive) = mpsc::sync_channel(1);
        if std::thread::Builder::new()
            .name("pals-bounded-actual-cuda-warm-check".into())
            .spawn(move || {
                let report = execute_request(
                    &args,
                    started,
                    deadline,
                    whole,
                    &worker_cancel,
                    &worker_capture,
                )
                .unwrap_or_else(|error| failure_report(error, started));
                let _ = send.send(report);
            })
            .is_err()
        {
            return publish(
                failure_report(fail("supervisor", "worker_spawn_failed"), started),
                Some(&output),
            );
        }
        let mut canceled = false;
        let report = match receive.recv_timeout(deadline.saturating_duration_since(Instant::now()))
        {
            Ok(report) => report,
            Err(_) => {
                canceled = true;
                cancel.store(true, Ordering::Release);
                match receive.recv_timeout(whole.saturating_duration_since(Instant::now())) {
                    Ok(mut report) => {
                        report["execution_checks_passed"] = json!(false);
                        report["passed"] = json!(false);
                        report
                    }
                    Err(_) => {
                        let mut report = failure_report(
                            fail("supervisor", "whole_deadline_physical_completion_unknown"),
                            started,
                        );
                        report["physical_completion"] =
                            json!("unknown_to_supervisor;_native_owner_not_joined");
                        report["native_owner_released"] = json!(false);
                        if cost_selected {
                            report["cost_same_seed_context"] = json!({"opt_in": true,
                                "execution_checks_passed": false, "benchmark_validation_complete": false,
                                "physical_completion": "unknown", "unreturned_lane_trace": "pending_unavailable_to_supervisor",
                                "same_seed_independent_reference_required": true, "comparison_status": "not_comparable",
                                "warm_over_fresh_api_wall_ratio": null, "warm": null, "fresh": null});
                        }
                        if let Ok(capture) = capture.try_lock() {
                            report["calls"] = json!(capture.wires());
                            report["actual_search_rechecks"] = json!(capture.rechecks);
                            report["scheduler_interrupted_stack_resume_after_gpu_repair"] =
                                capture.cpu_resume_evidence();
                            if cost_selected {
                                report["cost_same_seed_context"]["warm"] =
                                    json!(capture.cost_warm_lane);
                                report["cost_same_seed_context"]["original_accepted_repair_source"] =
                                    json!(capture.cost_fixed_repair.as_ref().map(|v| &v.source));
                                report["cost_same_seed_context"]["original_accepted_full_6144_seed_bits"] = json!(
                                    capture.cost_fixed_repair.as_ref().map(|v| &v.source_latent)
                                );
                            }
                        }
                        report
                    }
                }
            }
        };
        let mut report = report;
        report["supervisor_cancel_issued_at_work_deadline"] = json!(canceled);
        publish(report, Some(&output))
    }

    #[cfg(test)]
    mod tests {
        #[test]
        fn cuda_warm_load_diagnostic_keeps_original_typed_cause_and_static_guard() {
            use rz_eval::error::{BackendError, CauseCode, FailureKind, FailureStage};
            let original = BackendError::new(FailureKind::InvalidInput, FailureStage::Admission,
                "CUDA Warm observation payload limit exceeds the charged physical tensor reservation")
                .with_external_cause(CauseCode::OutputValidation, &"synthetic original cause");
            let expected_cause = original.cause.unwrap();
            let failure = cuda_warm_load_failure(NativeCudaWarmLoadError {
                stage: "observation_admission",
                cause: RoleError::InvalidOutput,
                backend: Some(Box::new(original.clone())),
            });
            assert_eq!(
                failure.load_cause.as_ref().unwrap().backend.as_deref(),
                Some(&original)
            );
            let wire = failure.wire();
            assert_eq!(wire["code"], "invalid_output");
            assert_eq!(wire["load_diagnostic"]["stage"], "observation_admission");
            assert_eq!(
                wire["load_diagnostic"]["backend"]["detail"],
                original.detail
            );
            assert_eq!(
                wire["load_diagnostic"]["backend"]["cause"]["prefix_sha256"],
                hex(&expected_cause.prefix_sha256)
            );
            assert_eq!(
                wire["load_diagnostic"]["backend"]["raw_native_text_published"],
                false
            );
            assert_eq!(
                wire["load_diagnostic"]["physical_run_or_completion_claim"],
                false
            );
            let raw = serde_json::to_string(&wire).unwrap();
            assert!(!raw.contains("synthetic original cause"));
            let legacy = fail("synthetic", "unchanged").wire();
            assert_eq!(legacy, json!({"stage":"synthetic","code":"unchanged"}));
        }
        use super::*;

        #[test]
        fn checker_failure_keeps_typed_cause_without_publishing_raw_error_text() {
            let original = PalsError::Checker(CheckerError::Unsupported(
                "paused stack shutdown owner observation",
            ));
            let failure = pals_failure("cleanup", "own_checker_shutdown_failed", original);
            assert!(matches!(
                failure.pals_cause.as_deref(),
                Some(PalsError::Checker(CheckerError::Unsupported(
                    "paused stack shutdown owner observation"
                )))
            ));
            let wire = failure.wire();
            assert_eq!(wire["code"], "own_checker_shutdown_failed");
            assert_eq!(wire["pals_diagnostic"]["kind"], "Checker");
            assert_eq!(wire["pals_diagnostic"]["checker"]["kind"], "Unsupported");
            assert_eq!(
                wire["pals_diagnostic"]["checker"]["detail"],
                "paused stack shutdown owner observation"
            );
            assert_eq!(wire["pals_diagnostic"]["raw_error_text_published"], false);
            let private_payload = "synthetic private backend path and native text";
            let wire = pals_failure(
                "search",
                "actual_search_failed",
                PalsError::Role(RoleError::Backend(private_payload.to_owned())),
            )
            .wire();
            assert_eq!(wire["pals_diagnostic"]["kind"], "Role");
            assert!(
                !serde_json::to_string(&wire)
                    .unwrap()
                    .contains(private_payload)
            );
        }

        #[test]
        fn role_failure_keeps_safe_enum_without_private_error_payload() {
            for (error, kind) in [
                (RoleError::Unavailable, "unavailable"),
                (RoleError::InvalidOutput, "invalid_output"),
                (RoleError::Canceled, "canceled"),
                (RoleError::Deadline, "deadline"),
                (
                    RoleError::PhysicalCompletionUnknown,
                    "physical_completion_unknown",
                ),
                (
                    RoleError::Backend("synthetic private path and error text".into()),
                    "backend_error",
                ),
            ] {
                let failure =
                    pals_failure("search", "actual_search_failed", PalsError::Role(error));
                assert!(matches!(
                    failure.pals_cause.as_deref(),
                    Some(PalsError::Role(_))
                ));
                let wire = failure.wire();
                assert_eq!(wire["pals_diagnostic"]["kind"], "Role");
                assert_eq!(wire["pals_diagnostic"]["role"]["kind"], kind);
                assert_eq!(
                    wire["pals_diagnostic"]["role"]["raw_error_text_published"],
                    false
                );
                assert!(
                    !serde_json::to_string(&wire)
                        .unwrap()
                        .contains("synthetic private")
                );
            }
        }

        #[test]
        fn observed_cpu_paused_owner_is_visible_at_admission_and_checker_shutdown() {
            use rz_search::cpu_checker::{CheckerReport, CpuChecker};
            let capture = Arc::new(Mutex::new(Capture::new(Instant::now())));
            capture.lock().unwrap().stage = "actual_pals_search";
            let engine = CpuEngine::with_resume_policy(
                CpuConfig {
                    profile: CpuProfile::Independent,
                    tt_entries: 128,
                    max_depth: 2,
                    quiescence_ply: 8,
                },
                CpuResumePolicy::PausedStack,
            )
            .unwrap();
            let observed = ObservedCpu {
                engine,
                capture: Arc::clone(&capture),
            };
            let mut checker = OwnedCpuChecker::new(observed).unwrap();
            let initial = checker.paused_stack_snapshot().unwrap();
            assert!(initial.complete && initial.owner_id != 0);
            assert!(!initial.admission_closed && !initial.owner_released);
            assert_eq!(initial, checker.paused_stack_snapshot().unwrap());
            // This is the same idle owner witness required before the first
            // PALS CPU task. No frames does not make an absent snapshot valid.
            assert!(!checker.discard_paused_stack());
            let idle = checker.paused_stack_snapshot().unwrap();
            assert_eq!(idle.owner_id, initial.owner_id);
            assert!(idle.complete);
            assert_eq!((idle.tokens_retained, idle.bytes_current), (0, 0));
            let cancel = AtomicBool::new(false);
            let position = Position::startpos();
            let limits = CpuLimits {
                max_depth: 2,
                max_nodes: 1,
                deadline: Some(Instant::now() + Duration::from_secs(5)),
            };
            let CheckerReport::Owned(partial) =
                checker.analyze(&position, limits, &cancel).unwrap()
            else {
                panic!("own CPU observer returned a foreign report");
            };
            let token = partial.resume.as_ref().unwrap();
            assert!(token.is_paused_stack() && checker.token_is_current(token));
            let paused = checker.paused_stack_snapshot().unwrap();
            assert_eq!(paused.tokens_retained, 1);
            assert!(paused.complete && paused.bytes_current > 0);
            let attempt = checker.last_attempt().unwrap().clone();
            let shutdown = checker.shutdown(Instant::now()).unwrap();
            assert!(shutdown.cleanup_complete);
            assert!(!shutdown.exit_observed && !shutdown.stdout_drained);
            let closed = checker.paused_stack_snapshot().unwrap();
            assert_eq!(closed.owner_id, initial.owner_id);
            assert!(closed.complete && closed.admission_closed && closed.owner_released);
            assert_eq!((closed.tokens_retained, closed.bytes_current), (0, 0));
            assert_eq!(closed.tokens_invalidated, paused.tokens_invalidated + 1);
            assert_eq!(closed.bytes_peak, paused.bytes_peak);
            assert!(!checker.token_is_current(token));
            assert_eq!(checker.last_attempt().unwrap().work, attempt.work);
            assert_eq!(checker.last_attempt().unwrap().elapsed, attempt.elapsed);
            assert!(checker.shutdown(Instant::now()).unwrap().cleanup_complete);
            assert_eq!(closed, checker.paused_stack_snapshot().unwrap());
            let trace = capture.lock().unwrap();
            assert_eq!(trace.cpu_attempts.len(), 1);
            assert_eq!(trace.cpu_attempts[0].wire["report"]["nodes"], partial.nodes);
            assert_eq!(
                trace.cpu_attempts[0].wire["report"]["pv"],
                json!(moves(&partial.pv))
            );
            assert!(trace.calls.is_empty());
            assert_eq!(trace.cpu_resume_evidence()["passed"], false);
        }

        #[test]
        fn conv_default_device_warm_selection_requires_capture_control_not_cost() {
            let mut args = Args {
                mode: "capture".into(),
                values: BTreeMap::new(),
                output: PathBuf::new(),
                cost_same_seed_context: false,
                cost_conv_default: false,
            };
            assert!(matches!(
                cuda_conv_selection(&args),
                Ok(PalsCuDnnConvSearch::LegacyImplicitExhaustive)
            ));
            args.values.insert(
                "--cudnn-conv-search".into(),
                "default-device-warm-v1".into(),
            );
            assert!(cuda_conv_selection(&args).is_err());
            for flag in [
                "--control-inventory",
                "--control-inventory-sha256",
                "--profile-root",
            ] {
                args.values.insert(flag.into(), "not-read".into());
            }
            assert!(matches!(
                cuda_conv_selection(&args),
                Ok(PalsCuDnnConvSearch::ExperimentalDefaultDeviceWarm)
            ));
            args.cost_same_seed_context = true;
            assert!(cuda_conv_selection(&args).is_err());
            args.cost_same_seed_context = false;
            args.mode = "verify-reference".into();
            assert!(cuda_conv_selection(&args).is_err());
            args.mode = "capture".into();
            args.values.insert(
                "--cudnn-conv-search".into(),
                "default-host-control-v1".into(),
            );
            assert!(cuda_conv_selection(&args).is_err());
        }

        #[test]
        fn publish_gate_preserves_capture_reference_and_partial_cost_exits() {
            let capture =
                json!({"phase": "capture", "passed": false, "execution_checks_passed": true});
            let reference = json!({"phase": "verify_reference", "passed": true, "execution_checks_passed": true});
            assert!(report_passes_execution_gate(&capture));
            assert!(report_passes_execution_gate(&reference));

            let cost = json!({"opt_in": true, "independent_same_seed_reference_passed": true,
                "benchmark_validation_complete": false, "comparison_status": "not_comparable",
                "warm_over_fresh_api_wall_ratio": null});
            let mut cost_reference = reference.clone();
            cost_reference["cost_same_seed_context"] = cost.clone();
            // Reproduce the missing success marker that previously exited 1.
            assert!(!report_passes_execution_gate(&cost_reference));
            let verified = completed_cost_reference_metadata(cost);
            cost_reference["cost_same_seed_context"] = verified.clone();
            assert!(report_passes_execution_gate(&cost_reference));
            assert_eq!(verified["benchmark_validation_complete"], false);
            assert_eq!(verified["comparison_status"], "not_comparable");
            assert!(verified["warm_over_fresh_api_wall_ratio"].is_null());
            let mut cost_capture = capture.clone();
            cost_capture["cost_same_seed_context"] = verified;
            assert!(report_passes_execution_gate(&cost_capture));

            for marker in [json!(false), Value::Null, json!("true")] {
                cost_reference["cost_same_seed_context"]["execution_checks_passed"] =
                    marker.clone();
                cost_capture["cost_same_seed_context"]["execution_checks_passed"] = marker;
                assert!(!report_passes_execution_gate(&cost_reference));
                assert!(!report_passes_execution_gate(&cost_capture));
            }
            for missing_or_failed in [
                json!({}),
                json!({"phase": "capture", "passed": false}),
                json!({"phase": "verify_reference", "passed": false, "execution_checks_passed": true}),
            ] {
                assert!(!report_passes_execution_gate(&missing_or_failed));
            }
        }
        #[test]
        fn cost_reference_receipt_distinguishes_known_zero_from_missing_consumer_and_fence() {
            let mut receipt = json!({"process_epoch": 1, "game_generation": 9,
                "quarantined": false, "physical_runs_in_flight": 0,
                "physically_completed_role_calls": 8, "completed_role_inputs": 8,
                "delivered_role_inputs": 8, "search_consumed_role_inputs": 0});
            assert!(cost_receipt_matches(&receipt, 8, 0, false));
            assert!(!cost_receipt_matches(&receipt, 8, 0, true));
            receipt["search_consumed_role_inputs"] = Value::Null;
            assert!(!cost_receipt_matches(&receipt, 8, 0, false));
            receipt["search_consumed_role_inputs"] = json!(0);
            receipt["physical_shutdown_confirmed"] = json!(true);
            receipt["native_buffers_released"] = json!(true);
            receipt["backend_stats"] = json!({"role_nn_runs_attempted": 10,
                "role_nn_runs_completed": 10, "role_nn_runs_failed_known": 0});
            assert!(cost_receipt_matches(&receipt, 8, 0, true));
            receipt["backend_stats"]["role_nn_runs_completed"] = Value::Null;
            assert!(!cost_receipt_matches(&receipt, 8, 0, true));
        }
        #[test]
        fn cost_opt_in_is_explicit_and_does_not_modify_ordinary_capture_event_bytes() {
            let mut enabled = false;
            assert!(cost_opt_in("verify-reference", &mut enabled).is_err());
            assert!(!enabled);
            assert!(cost_opt_in("capture", &mut enabled).is_ok());
            assert!(enabled);
            assert!(cost_opt_in("capture", &mut enabled).is_err());

            let mut wire = json!({"physical": {"dispatched": true, "ready": false}});
            let before = serde_json::to_vec(&wire).unwrap();
            observe_cost_elapsed(
                &mut wire,
                false,
                "dispatched_elapsed_ns",
                Duration::from_nanos(10),
            );
            assert_eq!(serde_json::to_vec(&wire).unwrap(), before);
            for (event, elapsed) in [
                ("dispatched_elapsed_ns", 10),
                ("terminal_elapsed_ns", 20),
                ("physical_callback_elapsed_ns", 30),
            ] {
                observe_cost_elapsed(&mut wire, true, event, Duration::from_nanos(elapsed));
                assert_eq!(wire["cost_observation"][event], elapsed);
            }
            // These observer times are metadata. A terminal observation alone
            // does not supply a known receipt, accepted consumer or CUDA Run.
            assert_eq!(wire["physical"]["ready"], false);
            assert!(wire.get("accepted").is_none());
            assert!(wire.get("execution").is_none());
        }
        #[test]
        fn seed_provenance_framing_preserves_raw_digest_and_signed_zero_bits() {
            let mut bits = vec![0.25f32.to_bits(); LATENT_ELEMENTS];
            bits[0] = (-0.0f32).to_bits();
            // Independent known hashes of 6144 little-endian FP32 bit words.
            assert_eq!(
                seed_provenance_digest(&bits),
                "31d746ae061e3380aa88d34b3a310197361ecec7205c4f35ff3b5a3367b642a5"
            );
            assert_eq!(
                bits_digest(&bits),
                "99ea5aff8cdd0b1a31d72ba623c983870d2316751d2af3b7f8541a0ebd838ea4"
            );
            bits[0] = 0.0f32.to_bits();
            assert_eq!(
                seed_provenance_digest(&bits),
                "2a90e9a61891c51937fa4465758740a3f4b41c0d5169a6045ada8963c29303d1"
            );
        }
        #[test]
        fn cost_metadata_keeps_the_same_repair_packet_full_seed_and_original_provenance() {
            let input = json!({"query": [1, 2], "records": [3, 4], "history_digest": "fixed"});
            let context =
                json!({"purpose": "RepairPolicy", "public_revision": 7, "game_generation": 1});
            let mut latent = vec![0.25f32.to_bits(); LATENT_ELEMENTS];
            latent[0] = (-0.0f32).to_bits();
            let seal = "a".repeat(64);
            let source = json!({"input_key_hex": "b".repeat(64)});
            let seed = json!({"source_input_hex": source["input_key_hex"],
                "latent_bits_digest_hex": seed_provenance_digest(&latent), "source_lease_id": 9,
                "seed_sequence": 4, "seal_hex": seal});
            let warm = json!({"input": input, "logical_context": context,
                "initial_latent_bits": latent, "warm_start": true,
                "seed_provenance": seed,
                "invocation": {"mode": "approx_cuda_warm_v2", "seed_seal_hex": seal}});
            let mut stable = None;
            assert!(fixed_cost_packet(
                &warm,
                &input,
                &context,
                &latent,
                &source,
                true,
                &mut stable
            ));
            assert!(fixed_cost_packet(
                &warm,
                &input,
                &context,
                &latent,
                &source,
                true,
                &mut stable
            ));
            for (field, replacement) in [
                (
                    "input",
                    json!({"query": [1, 3], "records": [3, 4], "history_digest": "fixed"}),
                ),
                (
                    "logical_context",
                    json!({"purpose": "ValueFresh", "public_revision": 7, "game_generation": 1}),
                ),
                (
                    "initial_latent_bits",
                    json!(vec![0.5f32.to_bits(); LATENT_ELEMENTS]),
                ),
                (
                    "seed_provenance",
                    json!({"source_input_hex": source["input_key_hex"],
                    "latent_bits_digest_hex": seed_provenance_digest(&latent), "source_lease_id": 10,
                    "seed_sequence": 4, "seal_hex": seal}),
                ),
            ] {
                let mut changed = warm.clone();
                changed[field] = replacement;
                assert!(!fixed_cost_packet(
                    &changed,
                    &input,
                    &context,
                    &latent,
                    &source,
                    true,
                    &mut stable
                ));
            }
            for invalid_digest in [
                bits_digest(&latent),
                // Wrong domain, wrong u64LE length and changed signed-zero bits.
                "1c5d99c251e746969f44af86c1e2fb08c8711e74af9ed2fa573d1113119b5a57".into(),
                "529debeb04a6b88b84089545051ac6b36b7923a3f9688de175e632ab432c8f6e".into(),
                "2a90e9a61891c51937fa4465758740a3f4b41c0d5169a6045ada8963c29303d1".into(),
            ] {
                let mut changed = warm.clone();
                changed["seed_provenance"]["latent_bits_digest_hex"] = json!(invalid_digest);
                // A new observation must reject malformed provenance itself.
                assert!(!fixed_cost_packet(
                    &changed, &input, &context, &latent, &source, true, &mut None
                ));
            }
            let fresh = json!({"input": input, "logical_context": context,
                "initial_latent_bits": vec![0f32.to_bits(); LATENT_ELEMENTS], "warm_start": false,
                "seed_provenance": null, "invocation": {"mode": "fresh"}});
            assert!(fixed_cost_packet(
                &fresh, &input, &context, &latent, &source, false, &mut None
            ));
            let mut wrong_purpose = fresh.clone();
            let value_context =
                json!({"purpose": "ValueFresh", "public_revision": 7, "game_generation": 1});
            wrong_purpose["logical_context"] = value_context.clone();
            assert!(!fixed_cost_packet(
                &wrong_purpose,
                &input,
                &value_context,
                &latent,
                &source,
                false,
                &mut None
            ));
            // Packet equality is source metadata only; no fixture Run, receipt,
            // worker join, acceptance or independent numerical pass is supplied.
        }
        #[test]
        fn cost_known_completion_and_zero_consumer_cannot_hide_unknown_or_accepted_work() {
            let before = json!({"physically_completed_role_calls": 7, "completed_role_inputs": 7,
                "delivered_role_inputs": 7, "search_consumed_role_inputs": 4,
                "private_warm_observation": {"accepted_seeds": 2}});
            let after = json!({"physically_completed_role_calls": 8, "completed_role_inputs": 8,
                "delivered_role_inputs": 8, "search_consumed_role_inputs": 4,
                "physical_runs_in_flight": 0, "quarantined": false,
                "private_warm_observation": {"accepted_seeds": 2, "pending_acceptance": false,
                    "pinned_entries": 0, "active_lease": null}});
            let call = json!({"physical": {"dispatched": true, "ready": true,
                "completed_ok": true, "completion_unknown": false},
                "delivered": true, "accepted": false, "rejected": "SearchFailedUnconsumed", "raw_output": {}});
            assert!(cost_known_unconsumed(&before, &after, &call));
            assert_eq!(
                cost_delta(&before, &after, "search_consumed_role_inputs"),
                Some(0)
            );
            let mut unknown = before.clone();
            unknown["search_consumed_role_inputs"] = Value::Null;
            assert_eq!(
                cost_delta(&unknown, &after, "search_consumed_role_inputs"),
                None
            );
            assert!(!cost_known_unconsumed(&unknown, &after, &call));
            let mut consumed = after.clone();
            consumed["search_consumed_role_inputs"] = json!(5);
            assert!(!cost_known_unconsumed(&before, &consumed, &call));
            let mut published = after.clone();
            published["private_warm_observation"]["accepted_seeds"] = json!(3);
            assert!(!cost_known_unconsumed(&before, &published, &call));
            let mut unknown_call = call.clone();
            unknown_call["physical"]["completion_unknown"] = json!(true);
            assert!(!cost_known_unconsumed(&before, &after, &unknown_call));
        }
        #[test]
        fn default_cost_is_explicit_env_zero_and_separate_from_generic_cost() {
            let mut args = Args {
                mode: "capture".into(),
                values: BTreeMap::new(),
                output: PathBuf::new(),
                cost_same_seed_context: true,
                cost_conv_default: false,
            };
            args.values.insert(
                "--cudnn-conv-search".into(),
                "default-device-warm-v1".into(),
            );
            for flag in [
                "--control-inventory",
                "--control-inventory-sha256",
                "--profile-root",
            ] {
                args.values.insert(flag.into(), "not-read".into());
            }
            assert!(cuda_conv_selection(&args).is_err());
            args.cost_conv_default = true;
            assert!(matches!(
                cuda_conv_selection(&args),
                Ok(PalsCuDnnConvSearch::ExperimentalDefaultDeviceWarm)
            ));
            assert!(cost_default_environment(&args, Some("0")).is_ok());
            for environment in [None, Some(""), Some("1"), Some("false")] {
                assert!(cost_default_environment(&args, environment).is_err());
            }
            args.values.remove("--cudnn-conv-search");
            assert!(cuda_conv_selection(&args).is_err());
            assert!(cost_default_environment(&args, Some("0")).is_err());
            args.cost_conv_default = false;
            assert!(matches!(
                cuda_conv_selection(&args),
                Ok(PalsCuDnnConvSearch::LegacyImplicitExhaustive)
            ));
            assert!(cost_default_environment(&args, None).is_ok());
        }

        #[test]
        fn default_cost_reference_namespace_cannot_be_legacy_or_automatic_tf32_claim() {
            let default = json!({"cost_profile": "same-seed-conv-default-env0/v1",
                "cudnn_conv_search_requested": "default-device-warm-v1",
                "nvidia_tf32_override_required": "0", "tf32_requested": false,
                "tf32_actual": "unknown", "tf32_retry": "unknown"});
            assert!(cost_profile_matches_reference(&default, &default));
            assert!(!cost_profile_matches_reference(&default, &json!({})));
            assert!(!cost_profile_matches_reference(
                &default,
                &json!({"cost_profile": "same-seed-legacy/v1"})
            ));
            let mut invented = default.clone();
            invented["tf32_actual"] = json!(false);
            assert!(!cost_profile_matches_reference(&invented, &default));
            assert!(cost_profile_matches_reference(&json!({}), &json!({})));
            assert!(!cost_profile_matches_reference(&json!({}), &default));
        }

        #[test]
        fn progress_cost_samples_keep_their_actual_origin_and_final_owner_totals_separate() {
            let cost = json!({"cost_execution_boundary": "resident_repair_actual_search_progress_before_later_P_publication"});
            let mut lane = json!({"integration_inputs_at_progress": 21,
                "receipt_at_resident_progress": {"process_epoch": 1, "game_generation": 1,
                    "quarantined": false, "physical_runs_in_flight": 0,
                    "physically_completed_role_calls": 21, "completed_role_inputs": 21,
                    "delivered_role_inputs": 21, "search_consumed_role_inputs": 21}});
            assert_eq!(cost_sample_origin(&cost, &lane, true, 40).ok(), Some(21));
            assert!(cost_sample_origin(&cost, &lane, true, 20).is_err());
            lane["receipt_at_resident_progress"]["search_consumed_role_inputs"] = json!(20);
            assert!(cost_sample_origin(&cost, &lane, true, 40).is_err());
            assert!(cost_sample_origin(&cost, &json!({}), true, 40).is_err());
            assert_eq!(
                cost_sample_origin(&json!({}), &json!({}), true, 40).ok(),
                Some(40)
            );
            assert_eq!(
                cost_sample_origin(&cost, &json!({}), false, 40).ok(),
                Some(0)
            );
        }

        #[test]
        fn cost_opt_in_has_finite_sample_and_forward_bounds_without_changing_default_wall() {
            assert_eq!(COST_PREPARATION_CALLS, 3);
            assert_eq!(COST_MEASURED_CALLS, 5);
            assert_eq!(COST_EXTRA_ROLE_FORWARDS, 18);
            assert_eq!(COST_DEVICE_BYTES_MAX, 397_312);
            let mut args = Args {
                mode: "capture".into(),
                values: BTreeMap::new(),
                output: PathBuf::from("metadata-fixture-only"),
                cost_same_seed_context: false,
                cost_conv_default: false,
            };
            assert_eq!(cost_work_wall(&args), Duration::from_secs(60));
            args.cost_same_seed_context = true;
            assert_eq!(cost_work_wall(&args), Duration::from_secs(45));
            args.cost_conv_default = true;
            assert_eq!(cost_work_wall(&args), Duration::from_secs(150));
            assert_eq!(cost_work_wall(&args) + CLEANUP, Duration::from_secs(180));
            args.cost_same_seed_context = false;
            assert_eq!(cost_work_wall(&args), Duration::from_secs(60));
        }

        #[test]
        fn cost_time_receipts_reject_mixed_deadline_namespaces() {
            let sample = |profile: &str, wall: u64| {
                json!({"wall_limit_ms": wall, "cleanup_limit_ms": 30_000,
                    "whole_limit_ms": wall + 30_000,
                    "cost_same_seed_context": {"cost_profile": profile,
                        "wall_limit_ms": wall, "cleanup_limit_ms": 30_000,
                        "whole_limit_ms": wall + 30_000}})
            };
            assert!(cost_declared_bound_matches_capture(&sample(
                "same-seed-legacy/v1",
                45_000,
            )));
            let default = sample("same-seed-conv-default-env0/v1", 150_000);
            assert!(cost_declared_bound_matches_capture(&default));
            assert!(!cost_declared_bound_matches_capture(&sample(
                "same-seed-conv-default-env0/v1",
                45_000,
            )));
            assert!(!cost_declared_bound_matches_capture(&sample(
                "same-seed-legacy/v1",
                150_000,
            )));
            assert!(!cost_declared_bound_matches_capture(&sample(
                "unknown/v1",
                150_000,
            )));
            for scope in ["capture", "cost"] {
                for field in ["wall_limit_ms", "cleanup_limit_ms", "whole_limit_ms"] {
                    let mut mixed = default.clone();
                    let target = if scope == "capture" {
                        &mut mixed
                    } else {
                        &mut mixed["cost_same_seed_context"]
                    };
                    target[field] = json!(1);
                    assert!(!cost_declared_bound_matches_capture(&mixed));
                }
            }
        }
        #[test]
        fn progress_replay_selects_exact_resident_repair_after_cpu_c_chain() {
            // Boundary fixtures only: these packets prove no NN execution,
            // acceptance, physical fence or real scheduler ownership.
            let call = |sequence, purpose, role| {
                json!({"request_id": {"epoch": 7, "sequence": sequence},
                    "stage": "actual_pals_search", "accepted": true,
                    "accepted_with_original_controls": true, "input": {"role": role},
                    "logical_context": {"purpose": purpose},
                    "physical": {"ready": true, "completed_ok": true, "completion_unknown": false}})
            };
            let proof = json!({"passed": true,
            "matched_actual_pause_repair_c_resume_sequences": [
                {"accepted_gpu_repair_request_id": {"epoch": 7, "sequence": 37}}
            ]});
            let mut calls = vec![
                call(37, "RepairPolicy", "proposer"),
                call(38, "ValueFresh", "proposer"),
                call(39, "ReplyPolicy", "critic"),
            ];
            assert_eq!(
                resident_repair_for_progress(calls.iter(), &proof),
                Some(RequestId {
                    epoch: rz_contracts::ProcessEpoch(7),
                    sequence: 37
                }),
            );
            // Reproduce the actual stale last_repair selection: a later P
            // publication replaces its slot even with a completed old chain.
            calls.push(call(51, "ProposePolicy", "proposer"));
            assert_eq!(resident_repair_for_progress(calls.iter(), &proof), None);
            calls.last_mut().unwrap()["accepted"] = json!(false);
            assert_eq!(
                resident_repair_for_progress(calls.iter(), &proof),
                Some(RequestId {
                    epoch: rz_contracts::ProcessEpoch(7),
                    sequence: 37
                }),
            );
        }

        #[test]
        fn progress_replay_refuses_unobserved_or_mismatched_actual_source() {
            let source = json!({"request_id": {"epoch": 7, "sequence": 37},
                "stage": "actual_pals_search", "accepted": true,
                "accepted_with_original_controls": true, "input": {"role": "proposer"},
                "logical_context": {"purpose": "RepairPolicy"},
                "physical": {"ready": true, "completed_ok": true, "completion_unknown": false}});
            let proof = json!({"passed": true,
            "matched_actual_pause_repair_c_resume_sequences": [
                {"accepted_gpu_repair_request_id": {"epoch": 7, "sequence": 37}}
            ]});
            for invalid in [
                json!({}),
                json!({"passed": false}),
                json!({"passed": "true"}),
                json!({"passed": true, "matched_actual_pause_repair_c_resume_sequences": []}),
                json!({"passed": true, "matched_actual_pause_repair_c_resume_sequences": [
                    {"accepted_gpu_repair_request_id": {"epoch": 8, "sequence": 37}}]}),
            ] {
                assert_eq!(
                    resident_repair_for_progress([&source].into_iter(), &invalid),
                    None
                );
            }
            for field in ["accepted", "accepted_with_original_controls"] {
                let mut invalid = source.clone();
                invalid[field] = json!(false);
                assert_eq!(
                    resident_repair_for_progress([&invalid].into_iter(), &proof),
                    None
                );
            }
            let mut invalid = source.clone();
            invalid["physical"]["completion_unknown"] = json!(true);
            assert_eq!(
                resident_repair_for_progress([&invalid].into_iter(), &proof),
                None
            );
            invalid = source.clone();
            invalid["stage"] = json!("same_question_record_revision_resume");
            assert_eq!(
                resident_repair_for_progress([&invalid].into_iter(), &proof),
                None
            );
        }

        fn resident_cpu_evidence_fixture() -> (Capture, RequestId) {
            // Synthetic receipt admission only; no NN, CPU search, live token
            // or real TaskTable execution is performed by this fixture.
            use rz_search::pals::store::{LineId, SituationId, StateId};
            let started = Instant::now();
            let deadline = started + Duration::from_secs(5);
            let position = Position::startpos();
            let mut capture = Capture::new(started);
            capture.stage = "actual_pals_search";
            capture.cpu_attempts.push(CpuAttemptCapture {
                state: position.position_identity(),
                deadline: Some(deadline),
                cancel_address: 7,
                wire: json!({"kind": "analyze", "stage": "actual_pals_search",
                    "returned_ok": true, "finished_elapsed_ns": 10,
                    "requested": {"max_depth": 3, "max_nodes": 8},
                    "report": {"completed_depth": 0,
                        "resume_token": {"is_paused_stack": true, "owner_current_after": true}}}),
            });
            capture.cpu_attempts.push(CpuAttemptCapture {
                state: position.position_identity(),
                deadline: Some(deadline),
                cancel_address: 7,
                wire: json!({"kind": "resume", "stage": "actual_pals_search",
                    "returned_ok": true, "started_elapsed_ns": 40,
                    "resume_source_attempt_index": 0, "input_token_current_after": false,
                    "requested": {"max_depth": 3, "max_nodes": 8},
                    "resume_token": {"is_paused_stack": true, "owner_current_before": true,
                        "retained_bytes": 128, "cumulative_work": {"nodes": 8}},
                    "report": {"nodes": 8, "completed_depth": 1}}),
            });
            capture
                .scheduler_resume_links
                .push(json!({"same_exact_task_key": true,
                "actual_resume_attempt_indices": [1]}));
            for (sequence, text, response, critic, repair_at, critic_at) in [
                (15, "e2e4", "e7e5", 33, 20, 30),
                (28, "d2d4", "d7d5", 44, 22, 32),
            ] {
                let movement = BoardMove::from_uci(text).unwrap();
                let reply = BoardMove::from_uci(response).unwrap();
                let id = RequestId {
                    epoch: rz_contracts::ProcessEpoch(7),
                    sequence,
                };
                let critic_id = RequestId {
                    sequence: critic,
                    ..id
                };
                let question = OwnedQuestion {
                    position: position.clone(),
                    legal: position.legal_moves(),
                    prefix: vec![movement],
                    proposal: vec![movement, reply],
                    counterexample: Some(vec![movement, reply]),
                    records: vec![],
                    context: RoleLogicalContext {
                        game_generation: 1,
                        search_generation: 1,
                        situation: SituationId {
                            slot: 0,
                            generation: 0,
                        },
                        state: StateId(0),
                        focus: LineId(sequence as usize),
                        purpose: RoleQueryPurpose::RepairPolicy,
                        prefix: vec![movement],
                        focus_sha256: [0; 32],
                        prefix_sha256: [0; 32],
                        proposal_sha256: [0; 32],
                        refutation_sha256: Some([0; 32]),
                        divergence_sha256: [0; 32],
                        public_revision: sequence,
                        situation_revision: 0,
                    },
                    deadline,
                };
                for (call_id, owned_question, at, purpose, role) in [
                    (id, Some(question), repair_at, "RepairPolicy", "proposer"),
                    (critic_id, None, critic_at, "ReplyPolicy", "critic"),
                ] {
                    capture.calls.push(CallCapture {
                        id: call_id,
                        wire: json!({"request_id": request_id(call_id), "accepted": true,
                            "accepted_with_original_controls": true, "stage": "actual_pals_search",
                            "accepted_context_elapsed_ns": at, "logical_context": {"purpose": purpose},
                            "input": {"role": role}, "physical": {"ready": true,
                                "completed_ok": true, "completion_unknown": false}}),
                        question: owned_question,
                        snapshot: position.snapshot(),
                        deadline,
                        cancel_address: 7,
                        raw: None,
                        accepted: true,
                    });
                }
                capture.rechecks.push(json!({"event": "finished",
                    "actual_search_continuation_completed": true,
                    "repaired": moves(&[movement, reply]),
                    "refutation": moves(&[movement, reply]),
                    "reply_context": {"game_generation": 1, "search_generation": 1},
                    "actual_new_prefix_c_tail_request_ids": [request_id(critic_id)]}));
            }
            (
                capture,
                RequestId {
                    epoch: rz_contracts::ProcessEpoch(7),
                    sequence: 28,
                },
            )
        }

        #[test]
        fn resident_cpu_proof_searches_past_first_historical_chain() {
            let (mut capture, resident) = resident_cpu_evidence_fixture();
            let historical = capture.cpu_resume_evidence();
            assert_eq!(historical["passed"], true);
            assert_eq!(
                historical["matched_actual_pause_repair_c_resume_sequences"][0]["accepted_gpu_repair_request_id"]
                    ["sequence"],
                15
            );
            let current = capture.cpu_resume_evidence_for_repair(Some(resident));
            assert_eq!(current["passed"], true);
            assert_eq!(
                resident_repair_for_progress(
                    capture.calls.iter().map(|call| &call.wire),
                    &historical
                ),
                None
            );
            assert_eq!(
                resident_repair_for_progress(capture.calls.iter().map(|call| &call.wire), &current),
                Some(resident)
            );
            capture.progress_replay_source = Some(resident);
            let final_owner = capture.cpu_resume_evidence();
            assert_eq!(
                final_owner["required_accepted_repair_request_id"],
                request_id(resident)
            );
            assert_eq!(
                final_owner["matched_actual_pause_repair_c_resume_sequences"][0]["accepted_gpu_repair_request_id"],
                request_id(resident)
            );
        }

        #[test]
        fn resident_cpu_proof_keeps_missing_c_owner_and_query_guards() {
            let (mut capture, resident) = resident_cpu_evidence_fixture();
            capture.rechecks[1]["actual_search_continuation_completed"] = json!(false);
            // As in the actual v18 failure, a valid historical CPU resume does
            // not grant a Warm replay of another resident Repair question.
            assert_eq!(capture.cpu_resume_evidence()["passed"], true);
            assert_eq!(
                capture.cpu_resume_evidence_for_repair(Some(resident))["passed"],
                false
            );
            capture.rechecks[1]["actual_search_continuation_completed"] = json!(true);
            capture.rechecks[1]["refutation"] = json!(["e2e4", "e7e5"]);
            assert_eq!(
                capture.cpu_resume_evidence_for_repair(Some(resident))["passed"],
                false
            );
            capture.rechecks[1]["refutation"] = json!(["d2d4", "d7d5"]);
            capture.cpu_attempts[1].wire["resume_token"]["owner_current_before"] = json!(false);
            assert_eq!(
                capture.cpu_resume_evidence_for_repair(Some(resident))["passed"],
                false
            );
        }

        #[test]
        fn existing_paused_stack_must_span_accepted_repair_c_and_actual_resume() {
            // Timestamp admission only: this fixture supplies no CUDA Run,
            // acceptance callback, live token, scheduler link or physical fence.
            assert!(paused_stack_spans_repair_and_c_tail(10, 20, 30, 40));
            for timestamps in [
                (25, 20, 30, 40), // A new pause after Repair is not the held stack.
                (20, 20, 30, 40),
                (10, 30, 20, 40),
                (10, 20, 20, 40),
                (10, 20, 30, 30),
                (10, 20, 40, 30),
            ] {
                assert!(!paused_stack_spans_repair_and_c_tail(
                    timestamps.0,
                    timestamps.1,
                    timestamps.2,
                    timestamps.3,
                ));
            }
        }
        #[test]
        fn cuda_warm_owner_limits_use_native_byte_units_and_refuse_missing_exclusive_budget() {
            let Ok(limits) = cuda_warm_limits("1", "397312") else {
                panic!("registered CUDA Warm owner limits were rejected");
            };
            assert_eq!(limits.max_leases, 1);
            assert_eq!(limits.device_bytes_max, 397312);
            let config = PalsModelConfig::full_line_interaction_v2();
            let maximum_tokens = config.public_memory_tokens(config.max_records).unwrap() as u64;
            assert_eq!(
                limits.device_bytes_max,
                maximum_tokens * config.kv_heads as u64 * config.head_dimension as u64 * 4 * 2 * 2
            );
            for (leases, bytes) in [
                ("0", "397312"),
                ("2", "397312"),
                ("1", "0"),
                ("1", ""),
                ("1", "+1"),
                ("1", "18446744073709551616"),
            ] {
                assert!(cuda_warm_limits(leases, bytes).is_err());
            }
        }
        #[test]
        fn profile_output_is_exclusive_without_changing_existing_input_canonicalization() {
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let root = std::env::temp_dir().join(format!(
                "rz-pals-warm-path-only-{}-{nonce}",
                std::process::id()
            ));
            std::fs::create_dir(&root).unwrap();
            let profile = root.join("profiles");
            assert!(external(&profile, true).is_ok());
            assert!(external(&profile, false).is_err());
            std::fs::create_dir(&profile).unwrap();
            assert!(external(&profile, false).is_ok());
            assert!(external(&profile, true).is_err());
            std::fs::remove_dir(&profile).unwrap();
            let output = root.join("capture.json");
            std::fs::write(&output, b"{}").unwrap();
            assert!(external(&output, true).is_err());
            assert!(external(&output, false).is_ok());
            std::fs::remove_file(&output).unwrap();
            #[cfg(unix)]
            {
                std::os::unix::fs::symlink(root.join("absent"), &profile).unwrap();
                assert!(external(&profile, true).is_err());
                std::fs::remove_file(&profile).unwrap();
            }
            std::fs::remove_dir(&root).unwrap();
        }
        #[test]
        fn independent_numeric_comparison_requires_complete_finite_role_outputs() {
            let raw = PalsRawOutput {
                candidate_logits: vec![0.25],
                wdl_logits: [0.1, 0.2, 0.3],
                divergence_logits: None,
                task_logits: None,
                private_latent: vec![0.5; LATENT_ELEMENTS],
            };
            assert!(compare_output(&raw, &raw).is_ok());
            let mut changed = raw.clone();
            changed.private_latent[6143] = 0.6;
            assert!(compare_output(&raw, &changed).is_err());
            changed.private_latent.pop();
            assert!(compare_output(&raw, &changed).is_err());
            changed = raw.clone();
            changed.wdl_logits[1] = f32::NAN;
            assert!(compare_output(&raw, &changed).is_err());
            changed = raw.clone();
            changed.divergence_logits = Some(vec![0.]);
            assert!(compare_output(&raw, &changed).is_err());
        }
        #[test]
        fn forward_budget_has_no_silent_retry_slot() {
            let counts = ForwardBudget::new(MAX_ROLE_FORWARDS);
            for _ in 0..MAX_ROLE_FORWARDS - STARTUP_ROLE_FORWARDS - FOLLOWUP_ROLE_FORWARDS {
                assert!(search_admission(&counts).is_ok());
            }
            for _ in 0..FOLLOWUP_ROLE_FORWARDS {
                assert!(admission(&counts).is_ok());
            }
            assert!(admission(&counts).is_err());
            assert!(search_admission(&counts).is_err());
            assert_eq!(counts.load(Ordering::Acquire), MAX_ROLE_FORWARDS);
            assert_ne!(bits_digest(&[0, 1]), bits_digest(&[1, 0]));
        }
        #[test]
        fn search_forwards_preserve_two_actual_followup_slots() {
            let counts = ForwardBudget::new(DEFAULT_ROLE_FORWARDS);
            for _ in 0..DEFAULT_ROLE_FORWARDS - STARTUP_ROLE_FORWARDS - FOLLOWUP_ROLE_FORWARDS {
                assert!(search_admission(&counts).is_ok());
            }
            assert!(search_admission(&counts).is_err());
            assert_eq!(
                counts.load(Ordering::Acquire),
                DEFAULT_ROLE_FORWARDS - FOLLOWUP_ROLE_FORWARDS
            );
            assert!(admission(&counts).is_ok());
            assert!(admission(&counts).is_ok());
            assert!(admission(&counts).is_err());
        }

        #[test]
        fn mid_search_followups_do_not_reserve_their_global_slots_twice() {
            let counts = ForwardBudget::new(DEFAULT_ROLE_FORWARDS);
            for _ in 0..25 {
                assert!(search_admission(&counts).is_ok());
            }
            // Counter-only replay of v19: real Warm request26/Value27 happened
            // here. This fixture performs no NN and grants no physical fence.
            assert!(admission(&counts).is_ok());
            assert!(admission(&counts).is_ok());
            for _ in 25..58 {
                assert!(search_admission(&counts).is_ok());
            }
            assert_eq!(counts.usage(Ordering::Acquire), (2, 58, 2));
            assert_eq!(counts.load(Ordering::Acquire), 62);
            assert!(search_admission(&counts).is_ok());
            assert!(search_admission(&counts).is_ok());
            assert_eq!(counts.usage(Ordering::Acquire), (2, 60, 2));
            assert_eq!(counts.load(Ordering::Acquire), DEFAULT_ROLE_FORWARDS);
            let full = counts.usage(Ordering::Acquire);
            assert!(search_admission(&counts).is_err());
            assert!(admission(&counts).is_err());
            assert_eq!(counts.usage(Ordering::Acquire), full);
        }

        #[test]
        fn unused_or_partially_failed_followups_keep_exact_reserved_quotas() {
            for charged_followups in 0..=FOLLOWUP_ROLE_FORWARDS {
                let counts = ForwardBudget::new(DEFAULT_ROLE_FORWARDS);
                for _ in 0..charged_followups {
                    // A charge remains spent if subsequent NN work fails; this
                    // budget has no refund or relabel-to-search operation.
                    assert!(admission(&counts).is_ok());
                }
                for _ in 0..60 {
                    assert!(search_admission(&counts).is_ok());
                }
                let before_failed_admission = counts.usage(Ordering::Acquire);
                assert!(search_admission(&counts).is_err());
                assert!(search_admission(&counts).is_err());
                assert_eq!(counts.usage(Ordering::Acquire), before_failed_admission);
                assert_eq!(before_failed_admission, (2, 60, charged_followups));
                for _ in charged_followups..FOLLOWUP_ROLE_FORWARDS {
                    assert!(admission(&counts).is_ok());
                }
                assert_eq!(counts.usage(Ordering::Acquire), (2, 60, 2));
                assert_eq!(counts.load(Ordering::Acquire), DEFAULT_ROLE_FORWARDS);
                assert!(admission(&counts).is_err());
            }
        }

        #[test]
        fn followup_reentry_cannot_borrow_unused_search_or_exceed_registered_caps() {
            // 46 is the existing cost namespace's integration allowance:
            // 64 total minus its independent 18-forward cost reservation.
            for maximum in [16, 46, DEFAULT_ROLE_FORWARDS, MAX_ROLE_FORWARDS] {
                let counts = ForwardBudget::new(maximum);
                assert!(admission(&counts).is_ok());
                assert!(admission(&counts).is_ok());
                let reserved = counts.usage(Ordering::Acquire);
                assert!(admission(&counts).is_err());
                assert_eq!(counts.usage(Ordering::Acquire), reserved);
                for _ in 0..maximum - STARTUP_ROLE_FORWARDS - FOLLOWUP_ROLE_FORWARDS {
                    assert!(search_admission(&counts).is_ok());
                }
                assert_eq!(counts.load(Ordering::Acquire), maximum);
                let full = counts.usage(Ordering::Acquire);
                assert!(search_admission(&counts).is_err());
                assert!(admission(&counts).is_err());
                assert_eq!(counts.usage(Ordering::Acquire), full);
            }
            for maximum in [STARTUP_ROLE_FORWARDS, MAX_ROLE_FORWARDS + 1] {
                let counts = ForwardBudget::new(maximum);
                let before = counts.usage(Ordering::Acquire);
                assert!(search_admission(&counts).is_err());
                assert!(admission(&counts).is_err());
                assert_eq!(counts.usage(Ordering::Acquire), before);
            }
        }

        #[test]
        fn independent_forward_lanes_charge_one_atomic_global_budget() {
            let counts = Arc::new(ForwardBudget::new(DEFAULT_ROLE_FORWARDS));
            std::thread::scope(|scope| {
                let search = Arc::clone(&counts);
                scope.spawn(move || {
                    for _ in 0..60 {
                        assert!(search_admission(&search).is_ok());
                    }
                    assert!(search_admission(&search).is_err());
                });
                let followup = Arc::clone(&counts);
                scope.spawn(move || {
                    assert!(admission(&followup).is_ok());
                    assert!(admission(&followup).is_ok());
                    assert!(admission(&followup).is_err());
                });
            });
            assert_eq!(counts.usage(Ordering::Acquire), (2, 60, 2));
            assert_eq!(counts.load(Ordering::Acquire), DEFAULT_ROLE_FORWARDS);
        }
        #[test]
        fn raw_tolerance_does_not_bypass_strict_legal_policy_or_wdl() {
            // Input preparation and decode are actual shared CPU APIs. This
            // numeric failure fixture executes no NN Session or CUDA request.
            let position = Position::startpos();
            let legal = position.legal_moves();
            let cancel = AtomicBool::new(false);
            let config = PalsModelConfig::full_line_interaction_v2();
            let input = prepare_role_input_for_config(
                &RoleQuery {
                    position: &position,
                    legal: &legal,
                    prefix: &[],
                    proposal: &[],
                    counterexample: None,
                    records: &[],
                    revision: 0,
                    deadline: Instant::now() + Duration::from_secs(5),
                    cancel: &cancel,
                },
                NativeQueryKind::Propose,
                [1; 32],
                &config,
            )
            .unwrap();
            let raw = PalsRawOutput {
                candidate_logits: vec![100.; legal.len()],
                wdl_logits: [100.; 3],
                divergence_logits: None,
                task_logits: None,
                private_latent: vec![0.; LATENT_ELEMENTS],
            };
            let mut changed = raw.clone();
            changed.candidate_logits[0] += 0.01;
            changed.wdl_logits[0] += 0.01;
            assert!(compare_output(&raw, &changed).is_ok());
            assert!(compare_decoded(&raw, &changed, &input, &config).is_err());
            let mut signed_zero = input.clone();
            signed_zero.query[7] = -0.;
            assert_ne!(input_f32_bits(&input), input_f32_bits(&signed_zero));
        }
        #[test]
        fn actual_cpu_pause_and_resume_cannot_substitute_for_gpu_repair_c_acceptance() {
            let capture = Arc::new(Mutex::new(Capture::new(Instant::now())));
            capture.lock().unwrap().stage = "actual_pals_search";
            let engine = CpuEngine::with_resume_policy(
                CpuConfig {
                    profile: CpuProfile::Independent,
                    tt_entries: 128,
                    max_depth: 2,
                    quiescence_ply: 8,
                },
                CpuResumePolicy::PausedStack,
            )
            .unwrap();
            let mut cpu = ObservedCpu {
                engine,
                capture: Arc::clone(&capture),
            };
            let position = Position::startpos();
            let cancel = AtomicBool::new(false);
            let limits = CpuLimits {
                max_depth: 2,
                max_nodes: 1,
                deadline: Some(Instant::now() + Duration::from_secs(5)),
            };
            let first = cpu.analyze(&position, limits, &cancel).unwrap();
            let token = first.resume.as_ref().unwrap();
            assert!(token.is_paused_stack());
            assert!(cpu.token_is_current(token));
            let resumed = cpu.resume(&position, token, limits, &cancel).unwrap();
            assert!(resumed.nodes > 0);
            assert!(!cpu.token_is_current(token));
            let capture = capture.lock().unwrap();
            assert_eq!(capture.cpu_attempts.len(), 2);
            assert_eq!(capture.cpu_attempts[0].wire["report"]["nodes"], first.nodes);
            assert_eq!(
                capture.cpu_attempts[1].wire["report"]["nodes"],
                resumed.nodes
            );
            assert_eq!(capture.cpu_attempts[1].wire["requested"]["max_nodes"], 1);
            assert_eq!(capture.cpu_resume_evidence()["passed"], false);
        }
    }
}
