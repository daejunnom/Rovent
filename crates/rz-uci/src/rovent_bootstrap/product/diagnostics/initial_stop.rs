//! Opt-in, Serialize-only companion to the unchanged lifecycle /1 row.
//! A stopped source is metadata, never a live admission or physical fence.

use super::*;
use crate::rovent_root_admission::source_seal::{InitialStopReason, StoppedInitial};
use crate::search_driver::DriverWork;

const DOMAIN: &str = "rz-rovent-initial-stop-source-observation/1";

#[derive(Serialize)]
pub(super) struct InitialStopSourceObservation {
    domain: &'static str,
    event: &'static str,
    lifecycle_sequence: u64,
    input_lock_sha256: String,
    implementation_sha256: String,
    mode: &'static str,
    authority: AuthorityObservation,
    receipt_domain: &'static str,
    source_kind: &'static str,
    stop_reason: &'static str,
    source_seal_sha256: String,
    state_and_history_sha256: String,
    rules_profile_sha256: String,
    root_budget_sha256: String,
    deadline_seal_sha256: String,
    work_source: &'static str,
    work: InitialWorkObservation,
    previous_clean_root_present: bool,
    used_original_startup_ready: bool,
    previous_clean_root: Option<PreviousRootObservation>,
    live_admission_issued: bool,
    physical_completion_attested: bool,
    capabilities_reissued: bool,
}

#[derive(Serialize)]
struct PreviousRootObservation {
    authority: AuthorityObservation,
    input_lock_sha256: String,
    root_admission_sha256: String,
}

#[derive(Serialize)]
struct InitialWorkObservation {
    root_entries: u64,
    total_node_visits: Option<u64>,
    checked_transitions: u64,
    native_nodes: Option<u64>,
    native_qsearch_nodes: Option<u64>,
    submitted_role_tickets: u64,
    consumed_role_outputs: u64,
    completed_nn_inputs: Option<u64>,
    consumed_nn_inputs: Option<u64>,
    completed_depth: Option<u16>,
    maximum_examined_ply: u16,
}

fn returned_work(report: &SearchSessionReport) -> Result<InitialWorkObservation, &'static str> {
    // Read this run's actual report. Do not call a zero-work constructor or
    // replace unknown values with zero in the diagnostic projection.
    let DriverWork::RoventNative {
        root_entries,
        total_node_visits,
        checked_transitions,
        native_nodes,
        native_qsearch_nodes,
        submitted_role_tickets,
        consumed_role_outputs,
        completed_nn_inputs,
        consumed_nn_inputs,
        completed_depth,
        maximum_examined_ply,
    } = &report.work
    else {
        return Err("initial_stop_original_work_variant_unavailable");
    };
    if *root_entries != 0
        || *total_node_visits != Some(0)
        || *checked_transitions != 0
        || *native_nodes != Some(0)
        || *native_qsearch_nodes != Some(0)
        || *submitted_role_tickets != 0
        || *consumed_role_outputs != 0
        || *completed_nn_inputs != Some(0)
        || *consumed_nn_inputs != Some(0)
        || completed_depth.is_some()
        || *maximum_examined_ply != 0
    {
        return Err("initial_stop_original_work_not_known_zero");
    }
    Ok(InitialWorkObservation {
        root_entries: *root_entries,
        total_node_visits: *total_node_visits,
        checked_transitions: *checked_transitions,
        native_nodes: *native_nodes,
        native_qsearch_nodes: *native_qsearch_nodes,
        submitted_role_tickets: *submitted_role_tickets,
        consumed_role_outputs: *consumed_role_outputs,
        completed_nn_inputs: *completed_nn_inputs,
        consumed_nn_inputs: *consumed_nn_inputs,
        completed_depth: *completed_depth,
        maximum_examined_ply: *maximum_examined_ply,
    })
}

fn reason(value: InitialStopReason) -> &'static str {
    match value {
        InitialStopReason::RootReplaced => "root_replaced",
        InitialStopReason::Canceled => "canceled",
        InitialStopReason::Deadline => "deadline",
    }
}

fn original_report<'a>(
    result: Result<&'a SearchSessionReport, &SearchSessionFailure>,
) -> Result<&'a SearchSessionReport, &'static str> {
    result.map_err(|_| "initial_stop_original_report_failed")
}

// Borrow the same driver/lifecycle scope for this projection only. This
// descriptor owns no source, report, clock or live admission capability.
struct InitialStopScope<'a> {
    context: &'a SearchSessionContext,
    lock: &'a rz_experiments::RoventInputLock,
    selected_mode: ExecutionMode,
    lifecycle_sequence: u64,
}

fn from_source(
    source: &StoppedInitial,
    report: &SearchSessionReport,
    scope: InitialStopScope<'_>,
    receipt_domain: &'static str,
    previous_clean_root: Option<PreviousRootObservation>,
) -> Result<InitialStopSourceObservation, &'static str> {
    let InitialStopScope {
        context,
        lock,
        selected_mode,
        lifecycle_sequence,
    } = scope;
    let expected_domain = match selected_mode {
        ExecutionMode::CpuPc | ExecutionMode::HybridPc => {
            super::super::super::pc_session::ROVENT_PC_INITIAL_DOMAIN
        }
        ExecutionMode::CpuPcr | ExecutionMode::HybridPcr => {
            super::super::super::nnuec_session::ROVENT_NNUEC_INITIAL_DOMAIN
        }
    };
    if receipt_domain != expected_domain || selected_mode != lock.manifest.engine.mode {
        return Err("initial_stop_original_receipt_family_unsupported");
    }
    let descriptor = source.descriptor();
    descriptor
        .verify_against(lock)
        .map_err(|_| "initial_stop_original_descriptor_mismatch")?;
    let authority = &descriptor.input.authority;
    if hex(source.startup_lock_digest()) != lock.canonical_sha256
        || authority.epoch != context.authority.epoch.0
        || authority.game != context.authority.game.0
        || authority.root != context.authority.root.0
        || authority.implementation_sha256 != hex(context.authority.implementation)
        || descriptor.input.state_and_history_sha256 != hex(source.state_and_history_digest())
        || descriptor.input.rules_profile_sha256 != hex(source.rules_profile_digest())
        || descriptor.root_budget_sha256 != hex(source.root_budget_digest())
        || descriptor.deadline_seal_sha256 != hex(source.deadline_seal_digest())
        || descriptor.digest().ok() != Some(source.digest())
        || report.best_move != source.fallback()
        || report.mate_strategy.is_some()
        || lifecycle_sequence == 0
    {
        return Err("initial_stop_original_source_scope_mismatch");
    }
    if let Some(previous) = &previous_clean_root {
        if previous.input_lock_sha256 != lock.canonical_sha256
            || previous.authority.epoch != context.authority.epoch.0
            || previous.authority.implementation_sha256 != hex(context.authority.implementation)
        {
            return Err("initial_stop_previous_clean_root_scope_mismatch");
        }
    }
    Ok(InitialStopSourceObservation {
        domain: DOMAIN,
        event: "stopped_before_live_admission",
        lifecycle_sequence,
        input_lock_sha256: hex(source.startup_lock_digest()),
        implementation_sha256: hex(context.authority.implementation),
        mode: mode(selected_mode),
        authority: context.authority.into(),
        receipt_domain,
        source_kind: "original_stopped_initial",
        stop_reason: reason(source.stop_reason()),
        source_seal_sha256: hex(source.digest()),
        state_and_history_sha256: hex(source.state_and_history_digest()),
        rules_profile_sha256: hex(source.rules_profile_digest()),
        root_budget_sha256: hex(source.root_budget_digest()),
        deadline_seal_sha256: hex(source.deadline_seal_digest()),
        work_source: "original_returned_search_session_report",
        work: returned_work(report)?,
        previous_clean_root_present: previous_clean_root.is_some(),
        used_original_startup_ready: previous_clean_root.is_none(),
        previous_clean_root,
        live_admission_issued: false,
        physical_completion_attested: false,
        capabilities_reissued: false,
    })
}

pub(super) fn project(
    product: &RoventNativeProductDriver,
    context: &SearchSessionContext,
    result: Result<&SearchSessionReport, &SearchSessionFailure>,
    sequence: u64,
) -> Result<InitialStopSourceObservation, &'static str> {
    let report = original_report(result)?;
    let lock = product.sources.lock();
    match product.inner.as_ref() {
        ActualProductDriver::Pc(driver) | ActualProductDriver::PcNnuec(driver) => {
            let receipt = driver
                .latest_initial()
                .map_err(|_| "initial_stop_original_receipt_read_failed")?
                .ok_or("initial_stop_original_receipt_unavailable")?;
            if receipt.domain() != super::super::super::pc_session::ROVENT_PC_INITIAL_DOMAIN
                || receipt.authority() != context.authority
                || receipt.input_lock_digest() != product.sources.lock_digest()
            {
                return Err("initial_stop_original_receipt_scope_mismatch");
            }
            let previous = receipt
                .previous_clean_root()
                .map(|root| PreviousRootObservation {
                    authority: root.authority.into(),
                    input_lock_sha256: hex(root.input_lock),
                    root_admission_sha256: hex(root.root_admission.digest()),
                });
            if receipt.used_original_startup_ready() != previous.is_none() {
                return Err("initial_stop_original_idle_custody_mismatch");
            }
            from_source(
                receipt.source(),
                report,
                InitialStopScope {
                    context,
                    lock,
                    selected_mode: product.sources.lock().manifest.engine.mode,
                    lifecycle_sequence: sequence,
                },
                receipt.domain(),
                previous,
            )
        }
        ActualProductDriver::Nnuec(driver) => {
            // In particular, never reuse latest_root's admitted-but-before-
            // frontier retirement as a stopped-before-admission source.
            let receipt = driver
                .latest_initial()
                .map_err(|_| "initial_stop_original_receipt_read_failed")?
                .ok_or("initial_stop_original_receipt_unavailable")?;
            if receipt.domain() != super::super::super::nnuec_session::ROVENT_NNUEC_INITIAL_DOMAIN
                || receipt.authority() != context.authority
                || receipt.input_lock_digest() != product.sources.lock_digest()
            {
                return Err("initial_stop_original_receipt_scope_mismatch");
            }
            let previous = receipt
                .previous_clean_root()
                .map(|root| PreviousRootObservation {
                    authority: root.authority.into(),
                    input_lock_sha256: hex(root.input_lock),
                    root_admission_sha256: hex(root.root_admission.digest()),
                });
            if receipt.used_original_startup_ready() != previous.is_none() {
                return Err("initial_stop_original_idle_custody_mismatch");
            }
            from_source(
                receipt.source(),
                report,
                InitialStopScope {
                    context,
                    lock,
                    selected_mode: product.sources.lock().manifest.engine.mode,
                    lifecycle_sequence: sequence,
                },
                receipt.domain(),
                previous,
            )
        }
        ActualProductDriver::Pcr(_) => Err("initial_stop_original_receipt_family_unsupported"),
    }
}

pub(super) fn write(
    output: &mut impl Write,
    row: &InitialStopSourceObservation,
    state: &mut State,
) -> Result<(), &'static str> {
    if state.first_issue.is_some() || state.scope_incomplete {
        return Err("initial_stop_prior_diagnostic_incomplete");
    }
    if state.initial_stop_rows >= MAX_ROOT_ROWS {
        return Err("initial_stop_row_limit");
    }
    let mut buffer = RowBuffer(Vec::new());
    buffer
        .0
        .try_reserve_exact(MAX_ROW_BYTES)
        .map_err(|_| "initial_stop_allocation_failed")?;
    serde_json::to_writer(&mut buffer, row).map_err(|_| "initial_stop_encoding_or_row_limit")?;
    buffer
        .write_all(b"\n")
        .map_err(|_| "initial_stop_row_limit")?;
    if buffer.0.len() > (MAX_STREAM_BYTES - MAX_ROW_BYTES).saturating_sub(state.bytes) {
        return Err("initial_stop_stream_byte_limit");
    }
    // Same shared stream and final-row reserve. An attempted partial write is
    // charged once and never retried or represented as complete evidence.
    state.bytes += buffer.0.len();
    state.initial_stop_rows += 1;
    output
        .write_all(&buffer.0)
        .map_err(|_| "initial_stop_write_failed")?;
    output.flush().map_err(|_| "initial_stop_flush_failed")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bridge::BuildSearchSettings;
    use crate::contracts::SessionScope;
    use crate::engine::{
        ProcessClock,
        driver_start_fixture::{FixtureStart, issue_start},
    };
    use crate::parser::GoLimits;
    use crate::rovent_root_admission::RoventRootAdmissionPolicy;
    use crate::rovent_root_admission::source_seal::{InitialAdmissionOutcome, SourceSeal};
    use rz_contracts::{GameGeneration, ProcessEpoch, RootGeneration};
    use rz_experiments::{
        ROVENT_EXECUTION_DOMAIN, ROVENT_POSITION_VISIT_UNIT, RoventExecutionManifest,
    };
    use rz_position::Position;
    use rz_search::time::TimeBudgetConfig;
    use serde_json::{Value, json};
    use std::{num::NonZeroU16, time::Duration};

    // Source/Rules/control unit fixture only. It issues the original test Start
    // and SourceSeal; no native model, evaluator, worker or live product starts.
    fn artifact(name: &str) -> Value {
        json!({"path":format!("fixture/{name}"),"sha256":"a".repeat(64),"bytes":32,
            "source":"https://example.org/initial-stop-source-fixture","license":"fixture-only"})
    }
    fn fixture(
        reason: InitialStopReason,
    ) -> (
        StoppedInitial,
        SearchSessionContext,
        rz_experiments::RoventInputLock,
    ) {
        let position = Position::startpos();
        let model = json!({"semantics":"fixture-pc","encoding":"fixture-encoding",
            "policy_head":"legal-candidates/1","value_head":"side-to-move-wdl/1",
            "implementation_sha256":"b".repeat(64),"weights":artifact("weights"),
            "provenance":artifact("provenance"),"diagnostic":false});
        let wire = json!({
            "domain":ROVENT_EXECUTION_DOMAIN,
            "contract_revision":rz_contracts::rovent::ROVENT_SEARCH_REVISION,
            "run_id":"initial-stop-companion-fixture","purpose":"validation",
            "engine":{"source_commit":"a".repeat(40),"source_components":{"fixture":"c".repeat(64)},
                "binary":artifact("binary"),"mode":"cpu_pc",
                "pc":{"proposer":model,"critic":model,"validator":null,
                    "backend":"rust_cpu","device":{"device":"cpu"},
                    "backend_configuration_sha256":"d".repeat(64),"runtime_artifacts":[],
                    "fp32":true,"tf32":false,"batch_width":1,
                    "ort_intra_threads":null,"ort_inter_threads":null,"ort_spinning":null},
                "researcher":null,"teachers":[],"nnuec":null,
                "search":{"node_unit":ROVENT_POSITION_VISIT_UNIT,"requested_nodes":7,
                    "depth_plies":2,"mate_moves":null,"registered_max_chess_plies":64,
                    "searchmoves":null,"search_configuration_sha256":"1".repeat(64),
                    "root_state_and_history_sha256":hex(rz_position::contracts::rules_snapshot_semantic_digest(&position.snapshot())),
                    "rules_profile_sha256":hex(rz_position::contracts::profile_digest())}},
            "resources":{"cpu_quota_millicpus":2000,
                "pools":[{"actors":["manager","pc_caller"],"max_threads":1,"cpu_charge_millicpus":1000}],
                "memory_high_bytes":6_u64<<30,"memory_max_bytes":12_u64<<30,
                "gpu_devices":0,"physical_nn_concurrency":1,"topology_policy":"shared-control/1"},
            "coordinator":{"policy_sha256":"2".repeat(64),"questions_max":64,
                "ancestor_depth_max":16,"waiting_jobs_max":16,"role_tickets_max":16,
                "cpu_credit_limit":0,"original_root_budget_sha256":"3".repeat(64),
                "original_deadline_seal_sha256":"4".repeat(64)},
            "wall_time_max_ms":60000,"cleanup_max_ms":1000,"output_bytes_max":1048576,
            "agent_renewable":true,"provenance":[artifact("source")]
        });
        let lock = RoventExecutionManifest::from_json(&wire.to_string())
            .unwrap()
            .lock()
            .unwrap();
        let clock = ProcessClock::new(ProcessEpoch(9501));
        let authority = SearchAuthority {
            epoch: clock.epoch(),
            game: GameGeneration(1),
            root: RootGeneration(1),
            implementation: Digest([9; 32]),
        };
        let go = GoLimits {
            nodes: Some(7),
            depth_plies: NonZeroU16::new(2),
            movetime_ms: (reason == InitialStopReason::Deadline).then_some(0),
            ..GoLimits::default()
        };
        let context = match issue_start(
            &position,
            go,
            authority,
            &clock,
            clock.runtime_clock().origin(),
            BuildSearchSettings {
                time_config: TimeBudgetConfig::default(),
                max_simulations: 4096,
                untimed_limit: Duration::from_secs(60),
            },
            Duration::from_millis(500),
        )
        .unwrap()
        {
            FixtureStart::Started(context) => context,
            FixtureStart::RulesOnly(_) => panic!("ongoing Rules must issue the actual test Start"),
        };
        let source = SourceSeal::prepare(
            &lock,
            RoventRootAdmissionPolicy::DynamicGo,
            &position,
            &context,
            &clock,
        )
        .unwrap();
        match reason {
            InitialStopReason::Canceled => context.control.cancel(),
            InitialStopReason::RootReplaced => {
                context.control.cancel();
                let mut replaced = context.authority;
                replaced.root = RootGeneration(2);
                *context.current.lock().unwrap() = SessionScope::Search(replaced);
            }
            InitialStopReason::Deadline => {}
        }
        let stopped = match source.into_initial().unwrap() {
            InitialAdmissionOutcome::StoppedInitial(stopped) => stopped,
            InitialAdmissionOutcome::Admitted(_) => panic!("stopped source became a live permit"),
        };
        (stopped, context, lock)
    }

    // This is an explicitly synthetic unit report used to exercise the
    // projection, not a native execution or an operational success receipt.
    fn report(source: &StoppedInitial) -> SearchSessionReport {
        SearchSessionReport {
            best_move: source.fallback(),
            work: DriverWork::RoventNative {
                root_entries: 0,
                total_node_visits: Some(0),
                checked_transitions: 0,
                native_nodes: Some(0),
                native_qsearch_nodes: Some(0),
                submitted_role_tickets: 0,
                consumed_role_outputs: 0,
                completed_nn_inputs: Some(0),
                consumed_nn_inputs: Some(0),
                completed_depth: None,
                maximum_examined_ply: 0,
            },
            mate_strategy: None,
        }
    }
    fn projected() -> InitialStopSourceObservation {
        let (source, context, lock) = fixture(InitialStopReason::Canceled);
        from_source(
            &source,
            &report(&source),
            InitialStopScope {
                context: &context,
                lock: &lock,
                selected_mode: ExecutionMode::CpuPc,
                lifecycle_sequence: 1,
            },
            super::super::super::super::pc_session::ROVENT_PC_INITIAL_DOMAIN,
            None,
        )
        .unwrap()
    }

    #[test]
    fn actual_stopped_source_projects_each_original_reason_and_all_seals() {
        for stop in [
            InitialStopReason::Canceled,
            InitialStopReason::RootReplaced,
            InitialStopReason::Deadline,
        ] {
            let (source, context, lock) = fixture(stop);
            let row = from_source(
                &source,
                &report(&source),
                InitialStopScope {
                    context: &context,
                    lock: &lock,
                    selected_mode: ExecutionMode::CpuPc,
                    lifecycle_sequence: 2,
                },
                super::super::super::super::pc_session::ROVENT_PC_INITIAL_DOMAIN,
                None,
            )
            .unwrap();
            assert_eq!(row.stop_reason, reason(source.stop_reason()));
            assert_eq!(
                row.state_and_history_sha256,
                hex(source.state_and_history_digest())
            );
            assert_eq!(row.rules_profile_sha256, hex(source.rules_profile_digest()));
            assert_eq!(row.root_budget_sha256, hex(source.root_budget_digest()));
            assert_eq!(row.deadline_seal_sha256, hex(source.deadline_seal_digest()));
            assert_eq!(row.source_seal_sha256, hex(source.digest()));
            assert_eq!(row.lifecycle_sequence, 2);
            assert!(!row.live_admission_issued && !row.physical_completion_attested);
            assert!(row.used_original_startup_ready && row.previous_clean_root.is_none());
        }
    }

    #[test]
    fn scope_and_wrong_initial_domain_cannot_be_retagged_as_stopped_source() {
        let (source, mut context, lock) = fixture(InitialStopReason::Canceled);
        let report = report(&source);
        assert!(
            from_source(
                &source,
                &report,
                InitialStopScope {
                    context: &context,
                    lock: &lock,
                    selected_mode: ExecutionMode::CpuPc,
                    lifecycle_sequence: 1,
                },
                "rz-rovent-nnuec-native-root/1",
                None,
            )
            .is_err()
        );
        context.authority.root = RootGeneration(9);
        assert!(
            from_source(
                &source,
                &report,
                InitialStopScope {
                    context: &context,
                    lock: &lock,
                    selected_mode: ExecutionMode::CpuPc,
                    lifecycle_sequence: 1,
                },
                super::super::super::super::pc_session::ROVENT_PC_INITIAL_DOMAIN,
                None,
            )
            .is_err()
        );
    }

    #[test]
    fn actual_work_projection_rejects_unknown_nonzero_and_other_variants() {
        let (source, _, _) = fixture(InitialStopReason::Canceled);
        let mut value = report(&source);
        assert!(returned_work(&value).is_ok());
        if let DriverWork::RoventNative { native_nodes, .. } = &mut value.work {
            *native_nodes = None;
        }
        assert!(returned_work(&value).is_err());
        let failure = SearchSessionFailure::debug("OriginalInitialFailure", &"unit failure");
        assert!(original_report(Err(&failure)).is_err());
    }

    #[test]
    fn wrong_lock_report_and_previous_custody_remain_unavailable() {
        let (source, context, mut lock) = fixture(InitialStopReason::Canceled);
        let mut value = report(&source);
        let domain = super::super::super::super::pc_session::ROVENT_PC_INITIAL_DOMAIN;
        value.best_move = None;
        assert!(
            from_source(
                &source,
                &value,
                InitialStopScope {
                    context: &context,
                    lock: &lock,
                    selected_mode: ExecutionMode::CpuPc,
                    lifecycle_sequence: 1,
                },
                domain,
                None,
            )
            .is_err()
        );
        value.best_move = source.fallback();
        let previous = PreviousRootObservation {
            authority: context.authority.into(),
            input_lock_sha256: "0".repeat(64),
            root_admission_sha256: hex(source.digest()),
        };
        assert!(
            from_source(
                &source,
                &value,
                InitialStopScope {
                    context: &context,
                    lock: &lock,
                    selected_mode: ExecutionMode::CpuPc,
                    lifecycle_sequence: 1,
                },
                domain,
                Some(previous),
            )
            .is_err()
        );
        lock.canonical_sha256 = "0".repeat(64);
        assert!(
            from_source(
                &source,
                &value,
                InitialStopScope {
                    context: &context,
                    lock: &lock,
                    selected_mode: ExecutionMode::CpuPc,
                    lifecycle_sequence: 1,
                },
                domain,
                None,
            )
            .is_err()
        );
        if let DriverWork::RoventNative { native_nodes, .. } = &mut value.work {
            *native_nodes = Some(1);
        }
        assert!(returned_work(&value).is_err());
        value.work = DriverWork::Cpu {
            nodes: 0,
            completed_depth: 0,
        };
        assert!(returned_work(&value).is_err());
    }

    #[test]
    fn companion_is_off_by_default_and_does_not_change_legacy_row_bytes() {
        let diagnostics = RoventLifecycleDiagnostics::new_from_sources(
            Digest([3; 32]),
            Digest([4; 32]),
            ExecutionMode::CpuPc,
            Arc::new(AtomicBool::new(false)),
        );
        assert!(!diagnostics.initial_stop_source.load(Ordering::Acquire));
        let state = State::default();
        let row = diagnostics.row(&state, "root", None, "original_current_root", None);
        let original = serde_json::to_vec(&row).unwrap();
        diagnostics.enable_initial_stop_source();
        assert!(diagnostics.initial_stop_source.load(Ordering::Acquire));
        assert_eq!(
            original,
            serde_json::to_vec(&diagnostics.row(
                &state,
                "root",
                None,
                "original_current_root",
                None
            ))
            .unwrap()
        );
    }

    #[test]
    fn companion_shares_stream_budget_but_not_root_counts_and_reserves_final_row() {
        let row = projected();
        let mut state = State {
            root_runs: 1,
            root_rows: 1,
            ..State::default()
        };
        let mut bytes = Vec::new();
        write(&mut bytes, &row, &mut state).unwrap();
        assert_eq!(state.bytes, bytes.len());
        assert_eq!(
            (state.root_runs, state.root_rows, state.initial_stop_rows),
            (1, 1, 1)
        );
        let decoded: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(decoded["domain"], DOMAIN);
        assert_eq!(
            decoded["work_source"],
            "original_returned_search_session_report"
        );
        assert_eq!(decoded["physical_completion_attested"], false);
        state.bytes = MAX_STREAM_BYTES - MAX_ROW_BYTES;
        assert_eq!(
            write(&mut bytes, &row, &mut state),
            Err("initial_stop_stream_byte_limit")
        );
        let diagnostics = RoventLifecycleDiagnostics::new_from_sources(
            Digest([3; 32]),
            Digest([4; 32]),
            ExecutionMode::CpuPc,
            Arc::new(AtomicBool::new(false)),
        );
        let final_row = diagnostics.row(
            &state,
            "session",
            None,
            "whole_session_finish_observation",
            None,
        );
        write_row(&mut bytes, &final_row, &mut state, true).unwrap();
        assert!(state.bytes <= MAX_STREAM_BYTES);
    }

    struct FailingWriter {
        fail_flush: bool,
        calls: usize,
    }
    impl Write for FailingWriter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.calls += 1;
            if self.fail_flush {
                Ok(bytes.len())
            } else {
                Err(io::Error::other("unit write failure"))
            }
        }
        fn flush(&mut self) -> io::Result<()> {
            Err(io::Error::other("unit flush failure"))
        }
    }

    #[test]
    fn io_failure_charges_one_attempt_without_retry_or_positive_repair() {
        for fail_flush in [false, true] {
            let mut state = State::default();
            let mut output = FailingWriter {
                fail_flush,
                calls: 0,
            };
            assert!(write(&mut output, &projected(), &mut state).is_err());
            assert_eq!(output.calls, 1);
            assert_eq!(state.initial_stop_rows, 1);
            assert!(state.bytes > 0 && state.bytes <= MAX_ROW_BYTES);
        }
        let mut row = projected();
        row.source_seal_sha256 = "a".repeat(MAX_ROW_BYTES);
        let mut state = State::default();
        let mut output = Vec::new();
        assert!(write(&mut output, &row, &mut state).is_err());
        assert!(output.is_empty());
        assert_eq!(state.bytes, 0);
    }
}
