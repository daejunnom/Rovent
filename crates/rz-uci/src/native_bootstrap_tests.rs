//! Pure parser and evidence-store checks. No assets or native runtime are loaded.
//! Completion metadata below is synthetic and does not establish NN capability.

use super::*;
use crate::{PositionPort, PositionSpec, engine::RulesUciPort};
use rz_position::PositionLimits;

const EPOCH: ProcessEpoch = ProcessEpoch(211);
const PRIVATE_MARKER: &str = "synthetic-private-asset-marker";

fn valid_arguments() -> Vec<String> {
    vec![
        "--onnx-cpu".into(),
        format!("--source-weights={PRIVATE_MARKER}/weights.pb.gz"),
        format!("--onnx-model={PRIVATE_MARKER}/model.onnx"),
        format!("--export-manifest={PRIVATE_MARKER}/manifest.json"),
        format!("--manifest-sha256={}", "ab".repeat(32)),
        format!("--ort-library={PRIVATE_MARKER}/runtime-library"),
        format!("--ort-sha256={}", "cd".repeat(32)),
        format!("--output-root={PRIVATE_MARKER}/output"),
    ]
}

#[test]
fn forwarded_source_profile_is_opt_in_and_keeps_existing_execution_guards() {
    let ordinary = NativeConfig::parse_with_source_profile(valid_arguments(), false).unwrap();
    let diagnostic = NativeConfig::parse_with_source_profile(valid_arguments(), true).unwrap();
    assert!(!ordinary.profiling_requested());
    assert!(diagnostic.profiling_requested());
    assert_eq!(ordinary.provider(), diagnostic.provider());
    assert_eq!(ordinary.manifest_sha256, diagnostic.manifest_sha256);
    assert_eq!(ordinary.ort_sha256, diagnostic.ort_sha256);
    assert_eq!(ordinary.search_simulations, diagnostic.search_simulations);
    assert_eq!(ordinary.final_move_policy, diagnostic.final_move_policy);
    let mut explicit = valid_arguments();
    explicit.push("--profile".into());
    assert!(NativeConfig::parse_with_source_profile(explicit, true).is_ok());
    for flag in ["--experimental-raw-cache", "--experimental-io-buffers"] {
        let mut arguments = valid_arguments();
        arguments.push(flag.into());
        assert!(NativeConfig::parse_with_source_profile(arguments, true).is_err());
    }
}

fn assert_private_config_rejection(arguments: Vec<String>) {
    let rejected = NativeConfig::parse(arguments).unwrap_err();
    assert!(matches!(rejected, NativeBootstrapError::Config(_)));
    for rendered in [rejected.to_string(), format!("{rejected:?}")] {
        assert!(!rendered.contains(PRIVATE_MARKER));
        assert!(rendered.len() <= 256, "parser errors must remain bounded");
    }
}

#[test]
fn native_search_budget_is_explicit_finite_and_default_compatible() {
    assert_eq!(
        NativeConfig::parse(valid_arguments())
            .unwrap()
            .engine_settings()
            .search
            .max_simulations,
        128
    );
    for limit in [1, 128, MAX_NATIVE_SIMULATIONS] {
        let mut args = valid_arguments();
        args.push(format!("--search-simulations={limit}"));
        let config = NativeConfig::parse(args).unwrap();
        assert_eq!(config.engine_settings().search.max_simulations, limit);
        assert_eq!(config.engine_settings().max_workers, 1);
    }
    for value in ["0", "4097", "-1", "unbounded"] {
        let mut args = valid_arguments();
        args.push(format!("--search-simulations={value}"));
        assert_private_config_rejection(args);
    }
    let mut args = valid_arguments();
    args.extend([
        "--search-simulations=1".into(),
        "--search-simulations=2".into(),
    ]);
    assert_private_config_rejection(args);
}

#[test]
fn explicit_tree_envelope_preserves_defaults_and_rejects_unbounded_or_duplicate_limits() {
    let original = NativeConfig::parse(valid_arguments())
        .unwrap()
        .engine_settings();
    assert_eq!(original.tree.max_edges, 100_000);
    for limit in [1, 262_144, MAX_NATIVE_TREE_EDGES] {
        let mut args = valid_arguments();
        args.push(format!("--search-max-edges={limit}"));
        let settings = NativeConfig::parse(args).unwrap().engine_settings();
        assert_eq!(settings.tree.max_edges, limit);
        assert_eq!(settings.tree.max_nodes, original.tree.max_nodes);
        assert_eq!(settings.tree.max_depth, original.tree.max_depth);
        assert_eq!(settings.search, original.search);
    }
    for limit in ["0", "4194305", "-1", "unbounded"] {
        let mut args = valid_arguments();
        args.push(format!("--search-max-edges={limit}"));
        assert_private_config_rejection(args);
    }
    let mut args = valid_arguments();
    args.extend([
        "--search-max-edges=262144".into(),
        "--search-max-edges=100000".into(),
    ]);
    assert_private_config_rejection(args);
}

#[test]
fn complete_native_config_keeps_private_paths_out_of_debug() {
    let config = NativeConfig::parse(valid_arguments()).unwrap();
    assert_eq!(config.manifest_sha256, [0xab; 32]);
    assert_eq!(config.ort_sha256, [0xcd; 32]);
    assert_eq!(
        config.source_weights,
        PathBuf::from(format!("{PRIVATE_MARKER}/weights.pb.gz"))
    );
    assert_eq!(
        config.output_root,
        PathBuf::from(format!("{PRIVATE_MARKER}/output"))
    );
    let public = format!("{config:?}");
    assert!(!public.contains(PRIVATE_MARKER));
    assert!(!public.contains("weights.pb.gz"));
    assert!(!public.contains("runtime-library"));
    assert!(public.len() <= 512);
}

#[test]
fn runtime_cache_root_is_optional_private_and_rejects_duplicate_selection() {
    assert!(
        NativeConfig::parse(valid_arguments())
            .unwrap()
            .runtime_cache_root
            .is_none()
    );
    let mut args = valid_arguments();
    args.push(format!("--runtime-cache-root={PRIVATE_MARKER}/cache"));
    let config = NativeConfig::parse(args.clone()).unwrap();
    assert_eq!(
        config.runtime_cache_root.unwrap(),
        PathBuf::from(format!("{PRIVATE_MARKER}/cache"))
    );
    args.push(format!("--runtime-cache-root={PRIVATE_MARKER}/other"));
    assert_private_config_rejection(args);
}

#[test]
fn exact_terminal_final_policy_is_explicit_and_keeps_s0_reproducible() {
    use rz_search::tree::FinalMovePolicy;
    assert_eq!(
        NativeConfig::parse(valid_arguments())
            .unwrap()
            .engine_settings()
            .final_move_policy,
        FinalMovePolicy::Visits
    );
    for (name, expected) in [
        ("visits", FinalMovePolicy::Visits),
        ("exact-terminal", FinalMovePolicy::ExactTerminal),
    ] {
        let mut args = valid_arguments();
        args.push(format!("--final-selection={name}"));
        assert_eq!(
            NativeConfig::parse(args)
                .unwrap()
                .engine_settings()
                .final_move_policy,
            expected
        );
    }
    let mut args = valid_arguments();
    args.push("--final-selection=neural-mate".into());
    assert_private_config_rejection(args);
    let mut args = valid_arguments();
    args.extend([
        "--final-selection=visits".into(),
        "--final-selection=exact-terminal".into(),
    ]);
    assert_private_config_rejection(args);
}

#[test]
fn attestation_is_explicit_optional_and_duplicate_selection_is_rejected() {
    assert!(
        !NativeConfig::parse(valid_arguments())
            .unwrap()
            .attestation_requested()
    );
    let mut selected = valid_arguments();
    selected.push("--attestation".into());
    assert!(
        NativeConfig::parse(selected.clone())
            .unwrap()
            .attestation_requested()
    );
    selected.push("--attestation".into());
    assert_private_config_rejection(selected);
    let mut named = valid_arguments();
    named.push("--attestation=true".into());
    assert_private_config_rejection(named);
}

#[test]
fn profile_is_opt_in_bounded_and_does_not_select_attestation() {
    let config = NativeConfig::parse(valid_arguments()).unwrap();
    assert!(!config.profiling_requested());
    assert!(config.source_journal().unwrap().is_none());
    let mut selected = valid_arguments();
    selected.push("--profile".into());
    let config = NativeConfig::parse(selected.clone()).unwrap();
    assert!(config.profiling_requested());
    assert!(!config.attestation_requested());
    assert_eq!(
        config
            .source_journal()
            .unwrap()
            .unwrap()
            .snapshot()
            .capacity,
        8192
    );
    selected.push("--profile".into());
    assert_private_config_rejection(selected);
    let mut named = valid_arguments();
    named.push("--profile=true".into());
    assert_private_config_rejection(named);
}

#[cfg(all(feature = "onnx-cuda", feature = "experimental-batch"))]
#[test]
fn batch_startup_keeps_loaded_provider_width_and_legacy_b1_boundary() {
    let cuda_args = || {
        let mut args = valid_arguments();
        args[0] = "--onnx-cuda".into();
        args.extend([
            format!("--cuda-bundle={PRIVATE_MARKER}/bundle.json"),
            format!("--cuda-bundle-sha256={}", "ef".repeat(32)),
            "--search-simulations=4096".into(),
            "--final-selection=visits".into(),
        ]);
        args
    };
    assert_eq!(
        NativeConfig::parse(cuda_args())
            .unwrap()
            .cuda_attestation_width(),
        Some(1)
    );
    for width in [1, 2, 4, 8, 16] {
        let mut args = cuda_args();
        args.extend([
            "--batch-attestation".into(),
            format!("--experimental-batch={width}"),
        ]);
        let config = NativeConfig::parse(args).unwrap();
        assert_eq!(config.cuda_attestation_width(), Some(width));
        assert!(config.batch_attestation_requested());
        assert!(!config.attestation_requested());
    }
    let mut unattested = cuda_args();
    unattested.push("--experimental-batch=4".into());
    assert!(
        NativeConfig::parse(unattested)
            .unwrap()
            .cuda_attestation_width()
            .is_none()
    );
    let mut legacy = cuda_args();
    legacy.extend(["--experimental-batch=4".into(), "--attestation".into()]);
    assert_private_config_rejection(legacy);
    for unsupported in [
        "--experimental-io-buffers",
        "--experimental-io-binding",
        "--experimental-cuda-graph",
    ] {
        let mut args = cuda_args();
        args.extend([
            "--batch-attestation".into(),
            "--experimental-batch=4".into(),
            unsupported.into(),
        ]);
        assert_private_config_rejection(args);
    }
}

#[test]
fn source_profile_rejects_unrepresented_runtime_modes_before_loading_assets() {
    for experiment in [
        "--experimental-raw-cache",
        "--experimental-batch=4",
        "--experimental-io-buffers",
        "--experimental-io-binding",
        "--experimental-cuda-graph",
    ] {
        let mut selected = valid_arguments();
        selected.extend(["--profile".into(), experiment.into()]);
        let failure = NativeConfig::parse(selected.clone()).unwrap_err();
        assert!(failure.to_string().contains("source profile v1 requires"));
        assert_private_config_rejection(selected);
    }
    let mut baseline = valid_arguments();
    baseline.extend([
        "--profile".into(),
        "--attestation".into(),
        "--experimental-batch=1".into(),
    ]);
    let baseline = NativeConfig::parse(baseline).unwrap();
    assert!(baseline.profiling_requested() && baseline.attestation_requested());

    // Defense below CLI: a direct C worker cannot issue B1 source records
    // from a multi-item or experimental binding/buffer backend either.
    let mut config = rz_eval::onnx::BackendConfig::cpu();
    config.max_batch = 1;
    assert!(config.source_profile_supported());
    config.max_batch = 4;
    assert!(!config.source_profile_supported());
    config.max_batch = 1;
    config.experiments.reuse_buffers = true;
    assert!(!config.source_profile_supported());
    config.experiments = rz_eval::onnx::ExecutionExperiments::default();
    config.experiments.io_binding = true;
    assert!(!config.source_profile_supported());
}

#[test]
fn native_parser_rejects_missing_duplicate_and_mixed_provider_arguments() {
    let valid = valid_arguments();
    for omitted in 0..valid.len() {
        let mut arguments = valid.clone();
        arguments.remove(omitted);
        assert_private_config_rejection(arguments);
    }
    for duplicate in &valid {
        let mut arguments = valid.clone();
        arguments.push(duplicate.clone());
        assert_private_config_rejection(arguments);
    }
    for mixed in [
        "--cpu-mock".to_owned(),
        "--mock-delay-ms=500".to_owned(),
        "--max-native-evaluations=129".to_owned(),
        format!("--unknown-provider={PRIVATE_MARKER}"),
        format!("--onnx-cpu={PRIVATE_MARKER}"),
    ] {
        let mut arguments = valid.clone();
        arguments.push(mixed);
        assert_private_config_rejection(arguments);
    }
}

#[test]
fn native_parser_bounds_hashes_and_path_bytes_without_echoing_bad_values() {
    for index in [4, 6] {
        for value in [
            String::new(),
            "a".repeat(63),
            "a".repeat(65),
            "CD".repeat(32),
            "cD".repeat(32),
            format!("{}g", "a".repeat(63)),
            PRIVATE_MARKER.to_owned(),
        ] {
            let mut arguments = valid_arguments();
            let name = arguments[index].split_once('=').unwrap().0;
            arguments[index] = format!("{name}={value}");
            assert_private_config_rejection(arguments);
        }
    }
    for index in [1, 2, 3, 5, 7] {
        for value in [
            String::new(),
            format!("{PRIVATE_MARKER}{}", "x".repeat(MAX_PATH_BYTES)),
            format!("{PRIVATE_MARKER}\nasset"),
            format!("{PRIVATE_MARKER}\0asset"),
            // UTF-8 byte length, rather than character count, is the boundary.
            "가".repeat(MAX_PATH_BYTES / 3 + 1),
        ] {
            let mut arguments = valid_arguments();
            let name = arguments[index].split_once('=').unwrap().0;
            arguments[index] = format!("{name}={value}");
            assert_private_config_rejection(arguments);
        }
        let mut arguments = valid_arguments();
        let name = arguments[index].split_once('=').unwrap().0;
        arguments[index] = format!("{name}={}", "x".repeat(MAX_PATH_BYTES));
        assert!(NativeConfig::parse(arguments).is_ok());
    }
}

struct Fixture {
    projection: ClassicalProjection,
    profile: EvaluatorProfile,
    position: PositionSnapshot<RulesState>,
    legal: LegalMoveView,
}

impl Fixture {
    fn new() -> Self {
        let owners = Arc::new(OwnerRegistry::default());
        let prepared = RulesUciPort::new(Arc::clone(&owners), PositionLimits::default())
            .prepare(&PositionSpec::default())
            .unwrap();
        let state = prepared.snapshot.state();
        let fill = HistoryFill::No;
        let binding = MaiaBinding::new(
            ModelHandle {
                owner: owners.allocate().unwrap(),
                slot: 0,
                generation: SlotGeneration(1),
                manifest: Digest([212; 32]),
            },
            EncodingHandle {
                owner: owners.allocate().unwrap(),
                slot: 0,
                generation: SlotGeneration(1),
                manifest: encoding_manifest(fill),
            },
            fill,
            Digest([213; 32]),
            1,
        )
        .unwrap();
        let profile = EvaluatorProfile {
            model: Arc::clone(binding.model()),
            precision: PrecisionProfile::Fp32,
            backend: binding.backend(),
            compute: ComputeBudget {
                min_steps: 1,
                max_steps: 1,
                require_full: true,
            },
            bytes: ByteBudget {
                host: HOST_BYTES_PER_ITEM,
                device: 0,
                pinned: 0,
            },
        };
        Self {
            projection: ClassicalProjection::new(binding),
            profile,
            position: state.snapshot().clone(),
            legal: state.legal_moves().clone(),
        }
    }

    fn request(&self, sequence: u64) -> EvalRequest<RulesState> {
        let context = EvalContext {
            revision: CONTRACT_REVISION,
            request: RequestId::new(EPOCH, sequence),
            selection: SelectionId::new(EPOCH, sequence),
            game: GameGeneration(1),
            root: RootGeneration(sequence),
            state: self.position.identity(),
            legal_order: self.legal.order(),
            input: self
                .projection
                .input_key(self.position.state(), self.legal.moves())
                .unwrap(),
            model: self.profile.model.handle(),
            encoding: self.profile.model.encoding().handle,
            precision: self.profile.precision,
            compute: self.profile.compute,
            backend: self.profile.backend,
        };
        EvalRequest::try_new(
            context,
            self.position.clone(),
            self.legal.clone(),
            Arc::clone(&self.profile.model),
            Deadline {
                clock: ClockDomain(EPOCH),
                at: MonotonicTick(1_000_000),
            },
            CancelToken::new(),
            self.profile.bytes,
        )
        .unwrap()
    }

    fn receipt(
        &self,
        sequence: u64,
        kind: NativeDiagnosticKind,
        code: ErrorCode,
        backend: Option<BackendError>,
    ) -> NativeDiagnosticReceipt {
        let request = self.request(sequence).context();
        NativeDiagnosticReceipt {
            ticket: (ExecutionId::new(EPOCH, sequence), request.request),
            context: CompletionContext {
                request,
                execution: (kind != NativeDiagnosticKind::DispatchRefused)
                    .then_some(ExecutionId::new(EPOCH, sequence)),
            },
            kind,
            failure: PhysicalFailure {
                contract: error(code, Stage::Backend, "synthetic native receipt"),
                backend,
            },
        }
    }

    fn report(&self) -> NativeRunReport {
        NativeRunReport::new(&self.profile, NativeWorkerOrigin::Injected).unwrap()
    }

    fn evidence(&self, origin: NativeWorkerOrigin) -> NativeEvidenceHandle {
        NativeEvidenceHandle(Arc::new(Mutex::new(EvidenceState {
            report: Some(NativeRunReport::new(&self.profile, origin).unwrap()),
            closed: None,
        })))
    }

    fn synthetic_output(&self, sequence: u64) -> EvalOutput {
        let request = self.request(sequence);
        EvalOutput {
            context: request.context(),
            legal: request.legal().clone(),
            policy: LegalPolicy::try_new(
                vec![1.0 / self.legal.moves().len() as f64; self.legal.moves().len()],
                1e-12,
            )
            .unwrap(),
            wdl: Wdl::try_new(0.4, 0.3, 0.3, 1e-6).unwrap(),
            viewpoint: Viewpoint::SideToMove,
            actual: ActualCompute {
                precision: PrecisionProfile::Fp32,
                steps: 1,
                full: true,
                backend: self.profile.backend,
                execution: Some(ExecutionId::new(EPOCH, sequence)),
                provenance: CacheProvenance::Computed,
            },
        }
    }
}

fn assert_same_receipt(actual: &NativeDiagnosticReceipt, expected: &NativeDiagnosticReceipt) {
    assert_eq!(actual.ticket, expected.ticket);
    assert_eq!(actual.context, expected.context);
    assert_eq!(actual.kind, expected.kind);
    assert_eq!(actual.failure.contract, expected.failure.contract);
    assert_eq!(actual.failure.backend, expected.failure.backend);
}

fn synthetic_backend(sequence: u64) -> BackendError {
    let raw = format!("{PRIVATE_MARKER}/native-run: original failure {sequence}");
    BackendError::new(
        FailureKind::BackendFailure,
        FailureStage::Backend,
        "synthetic physical worker failure",
    )
    .with_external_cause(CauseCode::OrtRun, &raw)
    .with_diagnostic(&format!("SyntheticNativeCode{sequence}"), &raw)
}

#[test]
fn normal_canceled_and_expired_audits_keep_counts_and_endpoint_receipts() {
    let fixture = Fixture::new();
    let mut report = fixture.report();
    for (code, offset) in [(ErrorCode::Canceled, 0), (ErrorCode::Expired, 100)] {
        for index in 1..=65 {
            assert!(
                report
                    .accept(fixture.receipt(
                        offset + index,
                        NativeDiagnosticKind::DispatchRefused,
                        code,
                        None,
                    ))
                    .is_none()
            );
        }
        let aggregate = if code == ErrorCode::Canceled {
            &report.canceled
        } else {
            &report.expired
        };
        assert_eq!(aggregate.count, 65);
        assert_same_receipt(
            aggregate.first.as_ref().unwrap(),
            &fixture.receipt(
                offset + 1,
                NativeDiagnosticKind::DispatchRefused,
                code,
                None,
            ),
        );
        assert_same_receipt(
            aggregate.last.as_ref().unwrap(),
            &fixture.receipt(
                offset + 65,
                NativeDiagnosticKind::DispatchRefused,
                code,
                None,
            ),
        );
    }
    assert!(!report.has_failure());
    assert!(report.failures.is_empty());
    assert!(report.overflow.is_none());
    assert_eq!(report.remaining_receipts(), MAX_FATAL_RECEIPTS + 1);
    assert_eq!(report.completed_by_runtime, 0);
    assert_eq!(report.origin, NativeWorkerOrigin::Injected);
    assert_eq!(
        report.model_manifest,
        fixture.profile.model.handle().manifest
    );
    assert_eq!(report.backend, fixture.profile.backend);
    let public = report.to_string();
    assert!(public.contains("canceled=65 expired=65"));
    assert!(public.contains("no full request journal"));
    assert!(public.len() <= 512);
}

#[test]
fn canceled_or_expired_with_backend_or_physical_kind_remains_fatal() {
    let fixture = Fixture::new();
    for code in [ErrorCode::Canceled, ErrorCode::Expired] {
        for (kind, backend) in [
            (
                NativeDiagnosticKind::DispatchRefused,
                Some(synthetic_backend(1)),
            ),
            (NativeDiagnosticKind::PhysicalFailure, None),
            (NativeDiagnosticKind::Quarantined, None),
        ] {
            let mut report = fixture.report();
            let receipt = fixture.receipt(1, kind, code, backend);
            assert_eq!(
                report.accept(receipt.clone()),
                Some(receipt.failure.contract)
            );
            assert!(report.has_failure());
            assert_eq!(report.canceled.count, 0);
            assert_eq!(report.expired.count, 0);
            assert_eq!(report.failures.len(), 1);
            assert_same_receipt(&report.failures[0], &receipt);
        }
    }
}

#[test]
fn fatal_store_retains_32_originals_and_the_first_overflow_with_safe_display() {
    let fixture = Fixture::new();
    let mut report = fixture.report();
    let expected: Vec<_> = (1..=33)
        .map(|sequence| {
            fixture.receipt(
                sequence,
                NativeDiagnosticKind::PhysicalFailure,
                ErrorCode::BackendFailure,
                Some(synthetic_backend(sequence)),
            )
        })
        .collect();
    for receipt in &expected {
        assert_eq!(
            report.accept(receipt.clone()),
            Some(receipt.failure.contract)
        );
    }
    assert_eq!(report.failures.len(), MAX_FATAL_RECEIPTS);
    for (actual, expected) in report.failures.iter().zip(&expected) {
        assert_same_receipt(actual, expected);
        let backend = actual.failure.backend.as_ref().unwrap();
        assert_eq!(backend.cause.unwrap().code, CauseCode::OrtRun);
        assert!(
            backend
                .native
                .as_ref()
                .unwrap()
                .message
                .contains(PRIVATE_MARKER)
        );
    }
    assert_same_receipt(report.overflow.as_ref().unwrap(), &expected[32]);
    assert_eq!(
        report.boundary_error.unwrap().code,
        ErrorCode::ResourceExhausted
    );
    assert_eq!(report.remaining_receipts(), 0);
    assert!(report.has_failure());
    let public = report.to_string();
    assert!(public.contains("fatal=32 overflow=true boundary=true"));
    assert!(public.contains("OrtRun"));
    assert!(!public.contains(PRIVATE_MARKER));
    assert!(!public.contains("original failure"));
    assert!(!format!("{report:?}").contains(PRIVATE_MARKER));
    assert!(public.len() <= 2048);
}

#[test]
fn normal_audit_count_overflow_retains_original_and_cannot_report_success() {
    let fixture = Fixture::new();
    for code in [ErrorCode::Canceled, ErrorCode::Expired] {
        let mut report = fixture.report();
        let first = fixture.receipt(1, NativeDiagnosticKind::DispatchRefused, code, None);
        assert!(report.accept(first.clone()).is_none());
        let aggregate = if code == ErrorCode::Canceled {
            &mut report.canceled
        } else {
            &mut report.expired
        };
        aggregate.count = u64::MAX;
        let rejected = fixture.receipt(2, NativeDiagnosticKind::DispatchRefused, code, None);
        let boundary = report.accept(rejected.clone()).unwrap();
        assert_eq!(boundary.code, ErrorCode::ResourceExhausted);
        assert_eq!(boundary.stage, Stage::Output);
        assert_eq!(report.boundary_error, Some(boundary));
        assert_same_receipt(report.overflow.as_ref().unwrap(), &rejected);
        let aggregate = if code == ErrorCode::Canceled {
            &report.canceled
        } else {
            &report.expired
        };
        assert_eq!(aggregate.count, u64::MAX);
        assert_same_receipt(aggregate.first.as_ref().unwrap(), &first);
        assert_same_receipt(aggregate.last.as_ref().unwrap(), &first);
        assert!(report.failures.is_empty());
        // The collector must not transfer another original after an overflow:
        // its next audit must not replace this first retained overflow receipt.
        assert_eq!(report.remaining_receipts(), 0);
        assert!(report.has_failure());
    }
}

#[test]
fn synthetic_completion_metadata_is_counted_only_for_the_native_profile() {
    let fixture = Fixture::new();
    let output = fixture.synthetic_output(1);
    let injected = fixture.evidence(NativeWorkerOrigin::Injected);
    injected.record_completed(&output).unwrap();
    let report = injected.try_take_report().unwrap().unwrap();
    assert_eq!(report.completed_by_runtime, 0);
    assert!(report.first_completed.is_none());
    assert!(report.last_completed.is_none());

    // Pure metadata acceptance; setting CpuOnnx here does not execute a backend.
    let synthetic = fixture.evidence(NativeWorkerOrigin::CpuOnnx);
    synthetic.record_completed(&output).unwrap();
    let last = fixture.synthetic_output(2);
    synthetic.record_completed(&last).unwrap();
    let report = synthetic.try_take_report().unwrap().unwrap();
    assert_eq!(report.completed_by_runtime, 2);
    assert_eq!(
        report.first_completed.unwrap().context.request,
        output.context
    );
    assert_eq!(report.last_completed.unwrap().context.request, last.context);
    assert_eq!(report.last_completed.unwrap().actual, last.actual);
    assert!(!report.has_failure());

    for invalid in 0..6 {
        let evidence = fixture.evidence(NativeWorkerOrigin::CpuOnnx);
        let mut altered = output.clone();
        match invalid {
            0 => altered.actual.precision = PrecisionProfile::Fp16,
            1 => altered.actual.steps = 2,
            2 => altered.actual.full = false,
            3 => altered.actual.provenance = CacheProvenance::ExactFeatureReuse,
            4 => altered.actual.backend = Digest([214; 32]),
            5 => altered.actual.execution = None,
            _ => unreachable!(),
        }
        let boundary = evidence.record_completed(&altered).unwrap_err();
        assert_eq!(boundary.code, ErrorCode::UnsupportedContract);
        assert_eq!(evidence.admission_error().unwrap(), Some(boundary));
        let report = evidence.try_take_report().unwrap().unwrap();
        assert_eq!(report.completed_by_runtime, 0);
        assert!(report.first_completed.is_none());
        assert!(report.last_completed.is_none());
        assert_eq!(report.boundary_error, Some(boundary));
        assert!(report.has_failure());
    }
}

#[test]
fn completed_count_overflow_closes_admission_and_keeps_last_original_metadata() {
    let fixture = Fixture::new();
    let evidence = fixture.evidence(NativeWorkerOrigin::CpuOnnx);
    let first = fixture.synthetic_output(1);
    evidence.record_completed(&first).unwrap();
    evidence
        .0
        .lock()
        .unwrap()
        .report
        .as_mut()
        .unwrap()
        .completed_by_runtime = u64::MAX;
    let last = fixture.synthetic_output(2);
    let boundary = evidence.record_completed(&last).unwrap_err();
    assert_eq!(boundary.code, ErrorCode::ResourceExhausted);
    assert_eq!(evidence.admission_error().unwrap(), Some(boundary));
    let report = evidence.try_take_report().unwrap().unwrap();
    assert_eq!(report.completed_by_runtime, u64::MAX);
    assert_eq!(
        report.first_completed.unwrap().context.request,
        first.context
    );
    assert_eq!(report.last_completed.unwrap().context.request, last.context);
    assert_eq!(report.last_completed.unwrap().actual, last.actual);
    assert_eq!(report.boundary_error, Some(boundary));
    assert!(report.has_failure());
}

#[test]
fn cuda_selection_requires_two_pins_and_rejects_mixed_or_mutable_profiles() {
    let mut valid = valid_arguments();
    valid[0] = "--onnx-cuda".into();
    valid.push(format!("--cuda-bundle={PRIVATE_MARKER}/bundle.json"));
    valid.push(format!("--cuda-bundle-sha256={}", "ef".repeat(32)));
    let parsed = NativeConfig::parse(valid.clone()).unwrap();
    assert_eq!(parsed.provider(), NativeProvider::Cuda);
    assert_eq!(parsed.cuda_bundle_sha256, Some([0xef; 32]));
    assert!(!format!("{parsed:?}").contains(PRIVATE_MARKER));
    for omitted in [valid.len() - 2, valid.len() - 1] {
        let mut arguments = valid.clone();
        arguments.remove(omitted);
        assert_private_config_rejection(arguments);
    }
    for forbidden in [
        "--onnx-cpu",
        "--device-id=1",
        "--arena-bytes=1",
        "--batch=2",
        "--tf32=true",
    ] {
        let mut arguments = valid.clone();
        arguments.push(forbidden.into());
        assert_private_config_rejection(arguments);
    }
    let mut cpu = valid.clone();
    cpu[0] = "--onnx-cpu".into();
    assert_private_config_rejection(cpu);
    let mut bad_hash = valid;
    *bad_hash.last_mut().unwrap() = format!("--cuda-bundle-sha256={}", "EF".repeat(32));
    assert_private_config_rejection(bad_hash);
}

#[test]
fn system_cuda_selection_is_explicit_and_never_accepts_legacy_receipts_or_cache() {
    let mut system = valid_arguments();
    system[0] = "--onnx-cuda".into();
    system.push(format!(
        "--system-cuda-lock={PRIVATE_MARKER}/system-lock.json"
    ));
    system.push(format!("--system-cuda-lock-sha256={}", "ef".repeat(32)));
    if cfg!(all(
        feature = "pals-adaptive",
        feature = "onnx-cuda",
        target_os = "linux"
    )) {
        let parsed = NativeConfig::parse(system.clone()).unwrap();
        assert!(parsed.system_cuda_lock.is_some());
        assert!(!format!("{parsed:?}").contains(PRIVATE_MARKER));
        let failure = NativeSessionFactory::load_parts(&OwnerRegistry::default(), &parsed)
            .err()
            .expect("unsupported system custody must fail before reading the fake assets");
        assert!(matches!(failure, NativeBootstrapError::Config(_)));
    } else {
        assert_private_config_rejection(system.clone());
    }
    for extra in [
        "--attestation",
        "--batch-attestation",
        "--cuda-bundle=old/bundle.json",
        "--cuda-bundle-sha256=efefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefef",
        "--runtime-cache-root=old/cache",
    ] {
        let mut arguments = system.clone();
        arguments.push(extra.into());
        assert_private_config_rejection(arguments);
    }
    for index in [system.len() - 2, system.len() - 1] {
        let mut arguments = system.clone();
        arguments.remove(index);
        assert_private_config_rejection(arguments);
    }
    let mut cpu = system;
    cpu[0] = "--onnx-cpu".into();
    assert_private_config_rejection(cpu);
}

#[test]
fn cpu_v1_typed_projection_rejects_cuda_origin_without_loading_native_assets() {
    let fixture = Fixture::new();
    let report = NativeRunReport::new(&fixture.profile, NativeWorkerOrigin::CudaOnnx).unwrap();
    let rejected = crate::native_attestation::RunEvidenceV1::try_from(&report).unwrap_err();
    assert_eq!(rejected.code, ErrorCode::UnsupportedContract);
    #[cfg(feature = "onnx-cuda")]
    for origin in [NativeWorkerOrigin::CpuOnnx, NativeWorkerOrigin::Injected] {
        let report = NativeRunReport::new(&fixture.profile, origin).unwrap();
        let rejected =
            crate::native_cuda_attestation::CudaRunEvidenceV1::try_from(&report).unwrap_err();
        assert_eq!(rejected.code, ErrorCode::UnsupportedContract);
    }
}

#[test]
fn guarded_search_metadata_separates_root_initialization_backup_and_d_return() {
    // These are synthetic receipts, not evidence of CUDA execution or capability.
    let fixture = Fixture::new();
    let evidence = fixture.evidence(NativeWorkerOrigin::CudaOnnx);
    let root = fixture.synthetic_output(1);
    let leaf = fixture.synthetic_output(2);
    let metadata = |output: &EvalOutput| rz_search::contracts::AcceptedEvaluation {
        context: CompletionContext {
            request: output.context,
            execution: output.actual.execution,
        },
        actual: output.actual,
    };
    evidence.record_completed(&root).unwrap();
    evidence.record_completed(&leaf).unwrap();
    evidence.record_search(&metadata(&root), 0).unwrap();
    evidence.record_search(&metadata(&leaf), 3).unwrap();
    let report = evidence.try_take_report().unwrap().unwrap();
    assert_eq!(report.completed_by_runtime, 2);
    assert_eq!(report.search_root_initializations.count, 1);
    assert_eq!(
        report
            .search_root_initializations
            .first
            .unwrap()
            .completed
            .context
            .request,
        root.context
    );
    assert_eq!(
        report
            .search_root_initializations
            .last
            .unwrap()
            .traversed_edges,
        0
    );
    assert_eq!(report.search_non_root_backups.count, 1);
    assert_eq!(
        report
            .search_non_root_backups
            .first
            .unwrap()
            .completed
            .actual,
        leaf.actual
    );
    assert_eq!(
        report.search_non_root_backups.last.unwrap().traversed_edges,
        3
    );
    let injected = fixture.evidence(NativeWorkerOrigin::Injected);
    injected.record_search(&metadata(&leaf), 1).unwrap();
    assert_eq!(
        injected
            .try_take_report()
            .unwrap()
            .unwrap()
            .search_non_root_backups
            .count,
        0
    );
}

#[test]
fn search_observer_overflow_preserves_last_original_and_closes_admission() {
    let fixture = Fixture::new();
    let evidence = fixture.evidence(NativeWorkerOrigin::CudaOnnx);
    let output = fixture.synthetic_output(2);
    let metadata = rz_search::contracts::AcceptedEvaluation {
        context: CompletionContext {
            request: output.context,
            execution: output.actual.execution,
        },
        actual: output.actual,
    };
    evidence
        .0
        .lock()
        .unwrap()
        .report
        .as_mut()
        .unwrap()
        .search_non_root_backups
        .count = u64::MAX;
    let rejected = evidence.record_search(&metadata, 1).unwrap_err();
    assert_eq!(evidence.admission_error().unwrap(), Some(rejected));
    let report = evidence.try_take_report().unwrap().unwrap();
    assert_eq!(report.search_non_root_backups.count, u64::MAX);
    assert_eq!(
        report
            .search_non_root_backups
            .last
            .unwrap()
            .completed
            .context,
        metadata.context
    );
    assert_eq!(
        report
            .search_non_root_backups
            .last
            .unwrap()
            .completed
            .actual,
        metadata.actual
    );
    assert!(report.has_failure());
}

#[test]
fn scheduler_and_final_delivery_losses_are_separate_and_never_count_as_nn_work() {
    let fixture = Fixture::new();
    let evidence = fixture.evidence(NativeWorkerOrigin::CudaOnnx);
    evidence
        .record_observations(NativeObservationReport {
            scheduler_events: 3,
            scheduler_dropped: 2,
            delivery_events: 5,
            delivery_dropped: 7,
            drain_discarded_results: 1,
            ..NativeObservationReport::default()
        })
        .unwrap();
    let report = evidence.try_take_report().unwrap().unwrap();
    assert_eq!(report.observations.scheduler_dropped, 2);
    assert_eq!(report.observations.delivery_dropped, 7);
    assert_eq!(report.observations.drain_discarded_results, 1);
    assert_eq!(report.completed_by_runtime, 0);
    assert_eq!(report.search_non_root_backups.count, 0);
    let evidence = fixture.evidence(NativeWorkerOrigin::CudaOnnx);
    let rejected = evidence
        .record_observations(NativeObservationReport {
            delivery_counter_overflow: true,
            ..NativeObservationReport::default()
        })
        .unwrap_err();
    assert_eq!(evidence.admission_error().unwrap(), Some(rejected));
    assert!(
        evidence
            .try_take_report()
            .unwrap()
            .unwrap()
            .observations
            .delivery_counter_overflow
    );
}
