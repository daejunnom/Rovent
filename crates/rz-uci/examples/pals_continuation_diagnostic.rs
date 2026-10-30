//! One registered actual V-free process/root, one fixed diagnostic recipe.
//! Register this example's real mapped image; no baseline image may be relabelled.
//! Stdout is a bounded raw observation, never Y1/T1/R1 acceptance. The parent
//! supervisor must retain original process-group/pipe/deadline/artifact guards.
#[cfg(not(feature = "pals-adaptive-diagnostics"))]
fn main() -> std::process::ExitCode {
    eprintln!("pals_continuation_diagnostic requires pals-adaptive-diagnostics");
    std::process::ExitCode::FAILURE
}

#[cfg(feature = "pals-adaptive-diagnostics")]
fn main() -> std::process::ExitCode {
    actual::main()
}

#[cfg(feature = "pals-adaptive-diagnostics")]
mod actual {
    use rz_contracts::ProcessEpoch;
    use rz_contracts::pals::RoleOutput;
    use rz_contracts::pals_adaptive::RoleTicketEnvelope;
    use rz_experiments::{PalsCoordinatorPolicyV5, PalsCpuSelectionV5};
    use rz_position::Position;
    use rz_search::pals::coordinator::{PhysicalRoleCompletion, RoleFailure};
    use rz_search::pals::engine::PostRepairRecheckPolicy;
    use rz_search::pals::review_context::ReviewView;
    use rz_uci::pals_adaptive::{
        AdaptiveCpuSliceDiagnostic, AdaptiveCpuSliceDiagnosticError,
        AdaptiveCpuSliceDiagnosticOptions, AdaptiveSessionManager, ContinuationDiagnostic,
        ContinuationDiagnosticOptions,
    };
    use rz_uci::pals_adaptive_bootstrap::ActualAdaptiveBackendV5;
    use rz_uci::pals_adaptive_bootstrap::diagnostic::{
        DeploymentDiagnosticObserversV5, prepare_deployment_diagnostic_actual_with_cancellation,
    };
    use rz_uci::pals_adaptive_startup::{
        PalsStartupOptionsV5, PinnedPalsStartupV5, parse_pals_search_settings_v5,
    };
    use rz_uci::pals_native::ticketed::{
        NativeTicketQuestion, TicketExecutionEvidence, TicketPreparation,
        VirtualReviewCollectionObserver, VirtualReviewPhysicalInvocationEvidence,
        VirtualReviewPreparedTicket,
    };
    use rz_uci::search_driver::SearchSessionDriver;
    use rz_uci::{Command, ParserLimits, PositionPort};
    use serde_json::json;
    use std::fmt::Debug;
    use std::io::{self, Write};
    use std::process::ExitCode;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::{Arc, Mutex, TryLockError};
    use std::time::{Duration, Instant};

    const RAW_BYTES_MAX: usize = 4 * 1024 * 1024;
    const HOT_RECORDS_MAX: usize = 64;
    const ROOT: Duration = Duration::from_secs(5);
    const Y1_FEN: &str = "3rk3/8/8/3p4/4P3/8/8/4K3 w - - 0 1";
    const EXTERNAL_OBSERVATION_SCAN_MAX: usize = 512;
    const EXTERNAL_REPORTS_MAX: usize = 64;
    const EXTERNAL_ROOT_BYTES_MAX: usize = 1024 * 1024;

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum Case {
        Y1Natural,
        T1Pending,
        R1Repair,
        ExternalRRepair,
    }
    impl Case {
        fn repair_observation(self) -> bool {
            matches!(self, Self::R1Repair | Self::ExternalRRepair)
        }
        fn checker(self) -> PalsCpuSelectionV5 {
            match self {
                Self::ExternalRRepair => PalsCpuSelectionV5::Stockfish19Full,
                _ => PalsCpuSelectionV5::Own,
            }
        }
        fn position(self) -> Result<Position, rz_position::PositionError> {
            match self {
                Self::Y1Natural | Self::ExternalRRepair => Position::from_fen(Y1_FEN),
                _ => Ok(Position::startpos()),
            }
        }
    }
    #[derive(Clone, Copy, Debug)]
    enum HotEvent {
        NativeSubmitted {
            ticket: RoleTicketEnvelope,
        },
        NativeCompleted {
            ticket: RoleTicketEnvelope,
            physical: PhysicalRoleCompletion,
            output_present: bool,
            evidence: Option<TicketExecutionEvidence>,
            invocation: Option<VirtualReviewPhysicalInvocationEvidence>,
        },
        NativeRetired {
            ticket: RoleTicketEnvelope,
            accepted_by_native: bool,
        },
        Dispatch {
            ticket: RoleTicketEnvelope,
            source_question: rz_contracts::Digest,
            source_view: rz_contracts::pals_adaptive::ReviewViewSeal,
            stage: rz_search::pals::coordinator::RefinementStage,
        },
        CpuSliceInstall(Result<(), AdaptiveCpuSliceDiagnosticError>),
    }
    impl HotEvent {
        fn wire(&self) -> serde_json::Value {
            match self {
                Self::NativeSubmitted { ticket } => json!({
                    "source":"NativeSubmitted","ticket":opaque(ticket)
                }),
                Self::NativeCompleted {
                    ticket,
                    physical,
                    output_present,
                    evidence,
                    invocation,
                } => json!({
                    "source":"NativeCompleted","ticket":opaque(ticket),"physical":opaque(physical),
                    "output_present":output_present,
                    "completed_nn_inputs":evidence.and_then(|e|e.completed_nn_inputs),
                    "measurement_complete":evidence.map(|e|e.measurement_complete),
                    "execution_evidence_debug":opaque(evidence),"invocation_debug":opaque(invocation)
                }),
                Self::NativeRetired {
                    ticket,
                    accepted_by_native,
                } => json!({
                    "source":"NativeRetired","ticket":opaque(ticket),"accepted_by_native":accepted_by_native
                }),
                Self::Dispatch {
                    ticket,
                    source_question,
                    source_view,
                    stage,
                } => json!({
                    "source":"SearchDispatch","ticket":opaque(ticket),"source_question":opaque(source_question),
                    "source_view":opaque(source_view),"stage":opaque(stage)
                }),
                Self::CpuSliceInstall(result) => {
                    json!({"source":"CpuSliceInstall","result_debug":opaque(result)})
                }
            }
        }
    }
    struct HotJournal {
        records: Mutex<Vec<HotEvent>>,
        unavailable: AtomicU64,
    }
    impl HotJournal {
        fn new() -> Result<Self, Box<dyn std::error::Error>> {
            let mut records = Vec::new();
            records.try_reserve_exact(HOT_RECORDS_MAX)?;
            Ok(Self {
                records: Mutex::new(records),
                unavailable: AtomicU64::new(0),
            })
        }
        fn record(&self, event: HotEvent) {
            match self.records.try_lock() {
                Ok(mut records) if records.len() < HOT_RECORDS_MAX => records.push(event),
                Ok(_) => {
                    self.unavailable.fetch_or(1, Ordering::AcqRel);
                }
                Err(TryLockError::WouldBlock) => {
                    self.unavailable.fetch_or(2, Ordering::AcqRel);
                }
                Err(TryLockError::Poisoned(_)) => {
                    self.unavailable.fetch_or(4, Ordering::AcqRel);
                }
            }
        }
        fn readback(&self) -> Result<Vec<HotEvent>, &'static str> {
            match self.records.try_lock() {
                Ok(records) => Ok(records.clone()),
                Err(TryLockError::WouldBlock) => Err("HotJournalBusy"),
                Err(TryLockError::Poisoned(_)) => Err("HotJournalPoisoned"),
            }
        }
    }
    struct NativeObserver(Arc<HotJournal>);
    impl VirtualReviewCollectionObserver for NativeObserver {
        fn submitted(
            &mut self,
            ticket: &RoleTicketEnvelope,
            _: &TicketPreparation<ReviewView, NativeTicketQuestion>,
            _: &VirtualReviewPreparedTicket,
        ) -> Result<(), RoleFailure> {
            self.0.record(HotEvent::NativeSubmitted { ticket: *ticket });
            Ok(())
        }
        fn completed(
            &mut self,
            ticket: &RoleTicketEnvelope,
            physical: PhysicalRoleCompletion,
            output: Option<&RoleOutput>,
            evidence: Option<TicketExecutionEvidence>,
            invocation: Option<VirtualReviewPhysicalInvocationEvidence>,
        ) -> Result<(), RoleFailure> {
            self.0.record(HotEvent::NativeCompleted {
                ticket: *ticket,
                physical,
                output_present: output.is_some(),
                evidence,
                invocation,
            });
            Ok(())
        }
        fn retired(
            &mut self,
            ticket: &RoleTicketEnvelope,
            accepted: bool,
        ) -> Result<(), RoleFailure> {
            self.0.record(HotEvent::NativeRetired {
                ticket: *ticket,
                accepted_by_native: accepted,
            });
            Ok(())
        }
    }
    fn fail(message: &'static str) -> Box<dyn std::error::Error> {
        io::Error::other(message).into()
    }
    fn opaque(value: &impl Debug) -> String {
        format!("{value:?}")
    }
    fn emit(value: &serde_json::Value) -> Result<(), Box<dyn std::error::Error>> {
        let bytes = serde_json::to_vec(value)?;
        if bytes.len() >= RAW_BYTES_MAX {
            return Err(fail("DiagnosticOutputCapacity"));
        }
        let mut output = io::stdout().lock();
        output.write_all(&bytes)?;
        output.write_all(b"\n")?;
        output.flush()?;
        Ok(())
    }

    // A single read after the original root fence, not a callback on every
    // search step. These are the Store's original foreign packets and public
    // records. Publication alone does not attest their semantic consumption;
    // the original continuation/role journals retain that separate boundary.
    fn external_root_observation(
        manager: &AdaptiveSessionManager<ActualAdaptiveBackendV5>,
    ) -> Result<serde_json::Value, &'static str> {
        use rz_search::pals::store::ObservationId;

        let root = manager
            .root_observation_snapshot()
            .map_err(|_| "OriginalRootFenceUnavailable")?;
        let port = manager
            .driver()
            .port()
            .ok_or("OriginalStoreOwnerUnavailable")?;
        let engine = port.engine();
        let stores = engine.stores();
        let issued = stores.observations.next_observation_sequence();
        if issued > EXTERNAL_OBSERVATION_SCAN_MAX || engine.records().len() > 128 {
            return Err("ExternalRootObservationCapacity");
        }
        let mut reports = Vec::new();
        for sequence in 0..issued {
            let observation = stores
                .observations
                .get(ObservationId(sequence))
                .map_err(|_| "OriginalObservationUnavailableNoImplicitColdLoad")?;
            let Some(report) = observation.external_report.as_ref() else {
                continue;
            };
            if reports.len() == EXTERNAL_REPORTS_MAX {
                return Err("ExternalReportObservationCapacity");
            }
            let state = stores
                .states
                .get(observation.state)
                .map_err(|_| "OriginalRulesAnchorUnavailable")?;
            let replay = state
                .uci_replay(256)
                .map_err(|_| "OriginalRulesHistoryUnavailable")?;
            let rules_digest = rz_position::contracts::rules_snapshot_semantic_digest(state)
                .0
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>();
            let checked_pv = observation
                .cpu_pv
                .map(|line| stores.lines.moves(line))
                .transpose()
                .map_err(|_| "OriginalRulesCheckedPvUnavailable")?;
            reports.push(json!({
                "observation_id":sequence,"state_id":observation.state.0,
                "store_execution_id":observation.execution.map(|id|id.0),
                "source":observation.source,"epoch":observation.epoch,
                "kind":opaque(&observation.kind),"scope":opaque(&observation.scope),
                "cpu_condition":observation.cpu_condition,
                "cpu_pv_line_id":observation.cpu_pv.map(|id|id.0),
                "rules_checked_pv":checked_pv.map(|line|line.iter().map(ToString::to_string).collect::<Vec<_>>()),
                "rules_anchor":{"fen":state.to_fen(),"semantic_sha256":rules_digest,
                    "history_origin":opaque(&replay.origin),"history_completeness":opaque(&replay.completeness),
                    "start_fen":replay.start_fen,"moves":replay.moves.iter().map(ToString::to_string).collect::<Vec<_>>()},
                "original_report":external_report_wire(report),
                "semantic_consumption":"NOT_INFERRED_FROM_STORE_PUBLICATION"
            }));
        }
        let records = engine.records().iter().map(|record| json!({
            "revision":record.revision,"parent_revision":record.parent_revision,
            "supersedes_revision":record.supersedes_revision,"origin_state_id":record.origin_state.0,
            "kind":opaque(&record.kind),"line":record.line.iter().map(ToString::to_string).collect::<Vec<_>>(),
            "cpu_observation_id":record.cpu_observation.map(|id|id.0),
            "perspective":opaque(&record.perspective)
        })).collect::<Vec<_>>();
        let observed = !reports.is_empty();
        let counts = root.manager_counters();
        let value = json!({
            "status":if observed{"OBSERVED"}else{"NOT_OBSERVED"},
            "scope":"ORIGINAL_STORE_AFTER_ORIGINAL_ROOT_FENCE_NO_RECONSTRUCTED_WORK",
            "authority":opaque(&root.authority()),"foreign_reports_counter":counts.foreign_reports,
            "foreign_reported_nodes_counter":counts.foreign_reported_nodes,
            "foreign_node_count_unknown_counter":counts.foreign_node_count_unknown,
            "reports":reports,"public_records":records,
            "case_acceptance":"NOT_ISSUED_BY_DIAGNOSTIC_RUNNER"
        });
        if serde_json::to_vec(&value)
            .map_err(|_| "ExternalRootSerialization")?
            .len()
            > EXTERNAL_ROOT_BYTES_MAX
        {
            return Err("ExternalRootBytesCapacity");
        }
        Ok(value)
    }

    fn external_report_wire(
        report: &rz_search::cpu_checker::ExternalCheckerReport,
    ) -> serde_json::Value {
        use rz_search::cpu_checker::ExternalRawScore;
        let score = match report.score {
            ExternalRawScore::Centipawns(cp) => json!({"kind":"Centipawns","value":cp}),
            ExternalRawScore::MateMoves(moves) => json!({"kind":"MateMoves","value":moves}),
            ExternalRawScore::Unknown => json!({"kind":"Unknown","value":null}),
        };
        json!({
            "identity":report.identity,"observed_uci":{"name":report.observed_uci.name,"author":report.observed_uci.author},
            "request_id":report.request_id,"best_move":report.best_move.map(|movement|movement.to_string()),
            "pv":report.pv.iter().map(ToString::to_string).collect::<Vec<_>>(),
            "score":score,"bound":opaque(&report.bound),"wdl_per_mille":report.wdl_per_mille,
            "perspective":opaque(&report.perspective),"requested_depth":report.requested_depth,
            "reported_depth":report.reported_depth,"seldepth":report.seldepth,"root_restricted":report.root_restricted,
            "completion":opaque(&report.completion),
            "work":{"nodes":report.work.nodes,"qnodes":report.work.qnodes,"tt_hits":report.work.tt_hits},
            "elapsed_ns":report.elapsed.as_nanos().to_string(),
            "scope":"ORIGINAL_FOREIGN_UNITS_NOT_OWN_RAW_WDL_OR_RULES_MATE_PROOF"
        })
    }
    type DiagnosticError = Box<dyn std::error::Error>;
    struct DiagnosticPosition {
        command: String,
        position: Position,
    }
    type DiagnosticArguments = (
        Case,
        u64,
        ProcessEpoch,
        Vec<String>,
        Option<DiagnosticPosition>,
    );

    fn diagnostic_position(command: &str) -> Result<DiagnosticPosition, DiagnosticError> {
        let Command::Position(spec) = rz_uci::parse(
            command,
            ParserLimits {
                max_line_bytes: 8192,
                ..ParserLimits::default()
            },
        )?
        else {
            return Err(fail("DiagnosticPositionCommandRequired"));
        };
        let port = rz_uci::engine::RulesUciPort::new(
            Arc::new(rz_uci::engine::OwnerRegistry::default()),
            rz_position::PositionLimits::default(),
        );
        let prepared = port.prepare(&spec)?;
        Ok(DiagnosticPosition {
            command: command.into(),
            position: prepared.snapshot.rules_position().clone(),
        })
    }

    // Custom input has its own raw domain; the original four default recipes
    // retain every existing raw field and source/limit/acceptance guard.
    fn record_custom_position(
        raw: &mut serde_json::Value,
        custom: &Option<(Case, serde_json::Value)>,
    ) {
        if let Some((case, input)) = custom {
            raw["domain"] = if *case == Case::ExternalRRepair {
                "rovezero.actual-external-continuation-diagnostic-raw/2"
            } else {
                "rovezero.actual-continuation-diagnostic-raw/6"
            }
            .into();
            raw["position_command"] = input["position_command"].clone();
            raw["position_input"] = input.clone();
        }
    }

    fn arguments() -> Result<DiagnosticArguments, DiagnosticError> {
        let args: Vec<String> = std::env::args().skip(1).collect();
        parse_arguments(args)
    }

    fn parse_arguments(args: Vec<String>) -> Result<DiagnosticArguments, DiagnosticError> {
        if !matches!(args.len(), 6 | 7) || args.iter().any(|s| s.len() > 8192) {
            return Err(fail("SixOrSevenBoundedArgumentsRequired"));
        }
        let mut case = None;
        let mut wall = None;
        let mut epoch = None;
        let mut position = None;
        let mut startup = Vec::with_capacity(3);
        for arg in args {
            if let Some(value) = arg.strip_prefix("--diagnostic-case=") {
                if case.is_some() {
                    return Err(fail("DuplicateCase"));
                }
                case = Some(match value {
                    "y1-natural" => Case::Y1Natural,
                    "t1-pending" => Case::T1Pending,
                    "r1-repair" => Case::R1Repair,
                    "external-r-repair" => Case::ExternalRRepair,
                    _ => return Err(fail("UnknownCase")),
                });
            } else if let Some(value) = arg.strip_prefix("--diagnostic-wall-ms=") {
                if wall.replace(value.parse::<u64>()?).is_some() {
                    return Err(fail("DuplicateWall"));
                }
            } else if let Some(value) = arg.strip_prefix("--diagnostic-epoch=") {
                if epoch.replace(value.parse::<u64>()?).is_some() {
                    return Err(fail("DuplicateEpoch"));
                }
            } else if let Some(value) = arg.strip_prefix("--diagnostic-position=") {
                if position.is_some() {
                    return Err(fail("DuplicateDiagnosticPosition"));
                }
                position = Some(diagnostic_position(value)?);
            } else {
                startup.push(arg);
            }
        }
        let wall = wall.ok_or_else(|| fail("WallRequired"))?;
        let epoch = epoch.ok_or_else(|| fail("EpochRequired"))?;
        if wall == 0 || wall > 60_000 || epoch == 0 || startup.len() != 3 {
            return Err(fail("FiniteOriginalWindowRequired"));
        }
        Ok((
            case.ok_or_else(|| fail("CaseRequired"))?,
            wall,
            ProcessEpoch(epoch),
            startup,
            position,
        ))
    }
    fn run(
        custom_input: &mut Option<(Case, serde_json::Value)>,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        let started = Instant::now();
        let (case, wall_ms, epoch, startup_args, position) = arguments()?;
        if let Some(position) = &position {
            *custom_input = Some((
                case,
                json!({"domain":"rovezero.diagnostic-exact-position-input/1",
                    "position_command":position.command,"rules_current_fen":position.position.to_fen(),
                    "history_origin":opaque(&position.position.history_origin()),
                    "history_completeness":opaque(&position.position.history_completeness()),
                    "known_history_len":position.position.known_history_len(),
                    "source":"PUBLIC_UCI_PARSER_AND_RULES_POSITION_PORT_BEFORE_NATIVE_STARTUP"}),
            ));
        }
        let until = started
            .checked_add(Duration::from_millis(wall_ms))
            .ok_or_else(|| fail("WindowOverflow"))?;
        let options = PalsStartupOptionsV5::parse(&startup_args)?;
        if options.metadata_preflight {
            return Err(fail("ActualOnly"));
        }
        let cwd = std::env::current_dir()?;
        let pinned = Arc::new(PinnedPalsStartupV5::open(options, &cwd)?);
        if wall_ms > pinned.lock.manifest.window.wall_time_max_ms {
            return Err(fail("OriginalWindowExceeded"));
        }
        let endpoint = &pinned.config.endpoint;
        if endpoint.cpu.cpu_r != case.checker()
            || endpoint.cpu.cpu_t != PalsCpuSelectionV5::Disabled
            || pinned
                .config
                .native_options
                .native_loading_profile
                .is_some()
            || pinned.config.native_options.cuda_device_id.is_some()
            || pinned.config.native_options.cuda_arena_bytes.is_some()
            || pinned.config.native_options.device_public_memory
            || (case == Case::ExternalRRepair && pinned.config.native_options.approximate_warm)
        {
            return Err(fail(if case == Case::ExternalRRepair {
                "ActualCpuExternalRNoTeacherProfileRequired"
            } else {
                "ActualCpuOwnNoTeacherProfileRequired"
            }));
        }
        let selected = parse_pals_search_settings_v5(&pinned.config, until)?;
        let (line, beam, role_max, cpu_max) = match case {
            Case::Y1Natural => (2, 1, 8, 32),
            Case::T1Pending => (2, 2, 8, 128),
            Case::R1Repair | Case::ExternalRRepair => (4, 1, 16, 1024),
        };
        let expected_policy = if matches!(case, Case::T1Pending) {
            PalsCoordinatorPolicyV5::CoordinatorAdaptive
        } else {
            PalsCoordinatorPolicyV5::CoordinatorFixed
        };
        if selected.config.line_plies != line
            || selected.config.beam_width != beam
            || selected.config.max_role_calls != role_max
            || selected.limits.max_cpu_nodes != cpu_max
            || selected.limits.cpu_depth != 2
            || endpoint.coordinator.policy != expected_policy
            || (matches!(case, Case::Y1Natural)
                && endpoint.coordinator.slices.slice_native_nodes_max != 1)
            || (matches!(case, Case::T1Pending) && endpoint.coordinator.slices.role_tickets_max < 2)
            || (case.repair_observation()
                && selected.post_repair_recheck == PostRepairRecheckPolicy::Disabled)
        {
            return Err(fail("PinnedCaseRecipeDiffersNoAutomaticRewrite"));
        }
        let drain = Duration::from_millis(pinned.config.native_options.drain_ms);
        let config_pin = pinned.config_sha256.clone();
        let registered_image = pinned.config.endpoint.binary.clone();
        let hot = Arc::new(HotJournal::new()?);
        let cpu = if matches!(case, Case::Y1Natural) {
            Some(AdaptiveCpuSliceDiagnostic::new(
                AdaptiveCpuSliceDiagnosticOptions { records_max: 64 },
            )?)
        } else {
            None
        };
        let continuation = if matches!(case, Case::Y1Natural) {
            None
        } else {
            Some(ContinuationDiagnostic::new(
                ContinuationDiagnosticOptions {
                    pending_snapshots_max: if matches!(case, Case::T1Pending) {
                        16
                    } else {
                        0
                    },
                    resolution_records_max: if case.repair_observation() { 64 } else { 0 },
                    role_consumption: case.repair_observation(),
                },
            )?)
        };
        let dispatch_hot = Arc::clone(&hot);
        let admission_hot = Arc::clone(&hot);
        let admission_cpu = cpu.clone();
        let external_root = (case == Case::ExternalRRepair)
            .then(|| Arc::new(Mutex::new(None::<Result<serde_json::Value, &'static str>>)));
        let closure_external = external_root.clone();
        let observers = DeploymentDiagnosticObserversV5 {
            native_observer: Box::new(NativeObserver(Arc::clone(&hot))),
            dispatch_observer: Arc::new(move |ticket, input, _| {
                dispatch_hot.record(HotEvent::Dispatch {
                    ticket: *ticket,
                    source_question: input.question_digest,
                    source_view: input.view_seal,
                    stage: input.stage,
                });
                Ok(())
            }),
            root_admission: Arc::new(move |manager| {
                if let Some(cpu) = &admission_cpu {
                    admission_hot.record(HotEvent::CpuSliceInstall(
                        manager.install_cpu_slice_diagnostic(cpu.clone()),
                    ));
                }
                Ok(())
            }),
            root_completion: Arc::new(move |manager, _| {
                if let Some(slot) = &closure_external {
                    if let Ok(mut slot) = slot.try_lock() {
                        *slot = Some(if slot.is_some() {
                            Err("ExternalRootReadbackRepeated")
                        } else {
                            external_root_observation(manager)
                        });
                    }
                }
                Ok(())
            }),
        };
        // One signal created before actual preparation and retained through all
        // original startup/cleanup stages. No callback resets cancellation.
        let startup_cancel = Arc::new(AtomicBool::new(false));
        let prepared = prepare_deployment_diagnostic_actual_with_cancellation(
            pinned,
            epoch,
            observers,
            until,
            Arc::clone(&startup_cancel),
        )?;
        let driver = prepared.driver();
        // All subsequent errors still call finish_process on this same owner.
        let mut authority = None;
        let mut run_elapsed = None;
        let execution = (|| -> Result<_, Box<dyn std::error::Error>> {
            if let Some(diagnostic) = &continuation {
                driver.install_continuation_diagnostic(diagnostic.clone())?;
            }
            let hard = Instant::now()
                .checked_add(ROOT)
                .ok_or_else(|| fail("RootOverflow"))?;
            let shutdown = hard
                .checked_add(drain)
                .ok_or_else(|| fail("RootDrainOverflow"))?;
            if shutdown > until || startup_cancel.load(Ordering::Acquire) {
                return Err(fail("OriginalWindowCannotFitOneRoot"));
            }
            let context = prepared.context(hard, shutdown)?;
            authority = Some(context.authority);
            let position = match position {
                Some(position) => position.position,
                None => case.position()?,
            };
            let run_started = Instant::now();
            let returned = driver.run(&position, &context, &mut |_| {});
            run_elapsed = Some(u64::try_from(run_started.elapsed().as_nanos())?);
            Ok(returned)
        })();
        // This performs original current-context retirement and original drain.
        // Physical unknown/quarantine remains on the retained owner on error.
        let process = prepared.finish_process(until);
        let completion = driver.completion_diagnostic();
        let cpu_snapshot = cpu.as_ref().map(|d| d.snapshot());
        let continuation_snapshot = continuation.as_ref().map(|d| d.snapshot());
        let hot_snapshot = hot.readback();
        let scalar_observations = scalar_wire::project(
            &prepared.clock(),
            authority,
            &hot_snapshot,
            &cpu_snapshot,
            &continuation_snapshot,
            continuation.as_ref().map(ContinuationDiagnostic::options),
        );
        let runtime_closure = scalar_wire::runtime_closure(&execution, &process, &completion);
        let external_snapshot = external_root.as_ref().map(|slot| match slot.try_lock() {
            Ok(slot) => match slot.as_ref() {
                Some(Ok(value)) => value.clone(),
                Some(Err(error)) => json!({"status":"UNAVAILABLE","reason":error}),
                None => json!({"status":"UNAVAILABLE","reason":"OriginalRootReadbackNotObserved"}),
            },
            Err(_) => json!({"status":"UNAVAILABLE","reason":"ExternalRootReadbackBusyOrPoisoned"}),
        });
        let available = hot.unavailable.load(Ordering::Acquire) == 0
            && hot_snapshot.is_ok()
            && authority.is_some()
            && completion.as_ref().is_ok_and(|c| {
                c.unavailable_reason.is_none()
                    && c.last
                        .as_ref()
                        .is_some_and(|last| Some(last.authority) == authority)
            })
            && cpu_snapshot.as_ref().is_none_or(|s| {
                s.as_ref()
                    .is_ok_and(|s| s.unavailable.is_none() && s.authority == authority)
            })
            && continuation_snapshot.as_ref().is_none_or(|s| {
                s.as_ref().is_ok_and(|s| {
                    s.unavailable_bits == 0
                        && s.authority == authority
                        && (!case.repair_observation()
                            || (s.role_install_observed == Some(true)
                                && s.role_readback_root_fence == Some(true)
                                && s.role_unavailable.is_none()))
                })
            })
            && external_snapshot
                .as_ref()
                .is_none_or(|snapshot| snapshot["status"] != "UNAVAILABLE");
        let successful_return = execution.as_ref().is_ok_and(|run| run.is_ok())
            && process
                .as_ref()
                .is_ok_and(|p| p.permits_archive_close() && p.store_archive_closed());
        let mut raw = json!({
            "domain":"rovezero.actual-continuation-diagnostic-raw/5",
            "case":opaque(&case), "case_acceptance":"NOT_ISSUED_BY_DIAGNOSTIC_RUNNER",
            "source_commit":env!("RZ_PALS_COMPILED_SOURCE_COMMIT"),
            "startup_config_sha256":config_pin,"registered_image":registered_image,
            "authority":authority.as_ref().map(opaque),"original_wall_ms":wall_ms,
            "root_hard_ms":ROOT.as_millis(),"run_elapsed_ns":run_elapsed,
            "scalar_observations":scalar_observations,
            "runtime_closure":runtime_closure,
            "driver_execution_debug":opaque(&execution),"completion_debug":opaque(&completion),
            "original_process_finish_debug":opaque(&process),
            "diagnostic_capture_available":available,"hot_unavailable_bits":hot.unavailable.load(Ordering::Acquire),
            "hot_records":hot_snapshot.as_ref().ok().map(|events|events.iter().map(HotEvent::wire).collect::<Vec<_>>()),
            "hot_readback_error":hot_snapshot.as_ref().err(),"cpu_slice_debug":opaque(&cpu_snapshot),
            "continuation_debug":opaque(&continuation_snapshot),
            "event_presence_verdict":"ROOT_INDEPENDENT_AUDIT_REQUIRED",
            "requested_yield_acceptance":"NOT_ISSUED",
            "external_signal_cancellation":"NOT_OBSERVED_BY_THIS_CONSUMER",
            "training_executed":false,"optimizer_steps":0
        });
        if let Some(external) = external_snapshot {
            // A distinct auxiliary domain preserves the original three raw/5
            // field sets and does not admit an external case through the Own
            // case auditor or the Collection teacher path.
            raw["domain"] = "rovezero.actual-external-continuation-diagnostic-raw/1".into();
            raw["position_command"] = format!("position fen {Y1_FEN}").into();
            raw["external_root"] = external;
        }
        record_custom_position(&mut raw, custom_input);
        emit(&raw)?;
        Ok(successful_return && available)
    }

    #[cfg(test)]
    mod external_tests {
        use super::*;
        use rz_position::Color;
        use rz_search::cpu_checker::{
            CheckerWork, ExternalBound, ExternalCheckerIdentity, ExternalCheckerReport,
            ExternalCompletion, ExternalModelMetadata, ExternalRawScore, ExternalTrainingKnowledge,
            ExternalUciIdentity,
        };
        use std::collections::BTreeMap;

        fn default_arguments(case: &str) -> Vec<String> {
            [
                "--startup-config=/tmp/not-opened-config.json".into(),
                "--startup-lock=/tmp/not-opened-lock.json".into(),
                format!("--startup-config-sha256={}", "a".repeat(64)),
                format!("--diagnostic-case={case}"),
                "--diagnostic-wall-ms=60000".into(),
                "--diagnostic-epoch=7".into(),
            ]
            .into()
        }

        #[test]
        fn diagnostic_position_rejects_non_position_illegal_history_and_unbounded_input() {
            for command in [
                "go nodes 1",
                "position startpos moves e2e5",
                "position fen bad",
            ] {
                let mut args = default_arguments("r1-repair");
                args.push(format!("--diagnostic-position={command}"));
                assert!(parse_arguments(args).is_err());
            }
            assert!(diagnostic_position(&"x".repeat(8193)).is_err());
            let mut duplicate = default_arguments("r1-repair");
            duplicate.extend([
                "--diagnostic-position=position startpos".into(),
                "--diagnostic-position=position startpos".into(),
            ]);
            assert!(parse_arguments(duplicate).is_err());
        }

        #[test]
        fn diagnostic_position_uses_public_rules_port_with_full_history() {
            let command = "position startpos moves e2e4 e7e5 g1f3 b8c6";
            let mut args = default_arguments("r1-repair");
            args.push(format!("--diagnostic-position={command}"));
            let (case, wall, epoch, startup, custom) = parse_arguments(args).unwrap();
            assert_eq!(case, Case::R1Repair);
            assert_eq!(wall, 60000);
            assert_eq!(epoch, ProcessEpoch(7));
            assert_eq!(startup.len(), 3);
            let custom = custom.unwrap();
            let mut expected = Position::startpos();
            for text in ["e2e4", "e7e5", "g1f3", "b8c6"] {
                expected.make_move(text.parse().unwrap()).unwrap();
            }
            assert_eq!(custom.command, command);
            assert_eq!(custom.position.to_fen(), expected.to_fen());
            assert_eq!(
                custom.position.known_history_len(),
                expected.known_history_len()
            );
            assert_eq!(custom.position.history_origin(), expected.history_origin());
            assert_eq!(
                custom.position.history_completeness(),
                expected.history_completeness()
            );
            assert_eq!(
                rz_uci::pals_native::pals_history_digest(&custom.position).unwrap(),
                rz_uci::pals_native::pals_history_digest(&expected).unwrap()
            );
        }

        #[test]
        fn diagnostic_default_positions_and_raw_domains_are_unchanged() {
            for (name, expected_fen) in [
                ("y1-natural", Y1_FEN.to_owned()),
                ("t1-pending", Position::startpos().to_fen()),
                ("r1-repair", Position::startpos().to_fen()),
                ("external-r-repair", Y1_FEN.to_owned()),
            ] {
                let (case, wall, _, _, custom) = parse_arguments(default_arguments(name)).unwrap();
                assert!(custom.is_none());
                assert_eq!(wall, 60000);
                assert_eq!(case.position().unwrap().to_fen(), expected_fen);
                let original = json!({"domain":if case == Case::ExternalRRepair {
                    "rovezero.actual-external-continuation-diagnostic-raw/1"
                } else {"rovezero.actual-continuation-diagnostic-raw/5"},
                    "unchanged_field":"original"});
                let mut raw = original.clone();
                record_custom_position(&mut raw, &None);
                assert_eq!(raw, original);
                let custom = Some((
                    case,
                    json!({"domain":"rovezero.diagnostic-exact-position-input/1",
                    "position_command":"position startpos"}),
                ));
                record_custom_position(&mut raw, &custom);
                assert_eq!(raw["unchanged_field"], "original");
                assert_eq!(raw["position_command"], "position startpos");
                assert_eq!(
                    raw["domain"],
                    if case == Case::ExternalRRepair {
                        "rovezero.actual-external-continuation-diagnostic-raw/2"
                    } else {
                        "rovezero.actual-continuation-diagnostic-raw/6"
                    }
                );
            }
        }

        fn report() -> ExternalCheckerReport {
            ExternalCheckerReport {
                identity: ExternalCheckerIdentity {
                    adapter_semantics: "explicit-test-only-not-actual".into(),
                    binary_sha256: "a".repeat(64),
                    launch_arguments_sha256: "b".repeat(64),
                    declared_name: "fixture".into(),
                    declared_version: "fixture".into(),
                    declared_source: "https://example.invalid/fixture".into(),
                    declared_license: "fixture".into(),
                    options: BTreeMap::from([("Threads".into(), "1".into())]),
                    assets: Vec::new(),
                    model_metadata: ExternalModelMetadata {
                        weights_sha256: None,
                        training: ExternalTrainingKnowledge::Unknown,
                        declared_rights: None,
                        precision: None,
                    },
                },
                observed_uci: ExternalUciIdentity {
                    name: "fixture-observed".into(),
                    author: None,
                },
                request_id: u64::MAX,
                best_move: Some("e2e4".parse().unwrap()),
                pv: vec!["e2e4".parse().unwrap(), "e7e5".parse().unwrap()],
                score: ExternalRawScore::MateMoves(-2),
                bound: ExternalBound::Lower,
                wdl_per_mille: None,
                perspective: Color::Black,
                requested_depth: 2,
                reported_depth: Some(1),
                seldepth: None,
                root_restricted: true,
                completion: ExternalCompletion::StoppedYield,
                work: CheckerWork {
                    nodes: Some(0),
                    qnodes: None,
                    tt_hits: None,
                },
                elapsed: Duration::from_nanos(123),
            }
        }

        #[test]
        fn external_case_never_changes_own_or_teacher_selection() {
            for own in [Case::Y1Natural, Case::T1Pending, Case::R1Repair] {
                assert_eq!(own.checker(), PalsCpuSelectionV5::Own);
            }
            assert_eq!(
                Case::ExternalRRepair.checker(),
                PalsCpuSelectionV5::Stockfish19Full
            );
            assert!(Case::ExternalRRepair.repair_observation());
            assert!(!Case::Y1Natural.repair_observation());
            assert!(!Case::T1Pending.repair_observation());
            assert!(Case::R1Repair.repair_observation());
            let position = Case::ExternalRRepair.position().unwrap();
            assert_eq!(position.to_fen(), Y1_FEN);
            assert!(!position.legal_moves().is_empty());
            assert!(position.legal_moves().len() < Position::startpos().legal_moves().len());
        }

        #[test]
        fn external_wire_keeps_original_request_pv_raw_units_and_unknowns() {
            let report = report();
            let wire = external_report_wire(&report);
            assert_eq!(wire["request_id"], u64::MAX);
            assert_eq!(wire["identity"]["binary_sha256"], "a".repeat(64));
            assert_eq!(wire["observed_uci"]["name"], "fixture-observed");
            assert!(wire["observed_uci"]["author"].is_null());
            assert_eq!(wire["pv"], json!(["e2e4", "e7e5"]));
            assert_eq!(wire["score"], json!({"kind":"MateMoves","value":-2}));
            assert_eq!(wire["bound"], "Lower");
            assert_eq!(wire["perspective"], "Black");
            assert_eq!(wire["completion"], "StoppedYield");
            assert_eq!(wire["work"]["nodes"], 0);
            assert!(wire["work"]["qnodes"].is_null());
            assert!(wire["wdl_per_mille"].is_null());
            assert_eq!(wire["elapsed_ns"], "123");
            assert!(wire.get("rules_mate_proof").is_none());
            assert!(wire.get("own_raw").is_none());
            let mut unknown = report.clone();
            unknown.score = ExternalRawScore::Unknown;
            assert_eq!(
                external_report_wire(&unknown)["score"],
                json!({"kind":"Unknown","value":null})
            );
            unknown.score = ExternalRawScore::Centipawns(-17);
            assert_eq!(
                external_report_wire(&unknown)["score"],
                json!({"kind":"Centipawns","value":-17})
            );
            assert_eq!(report.score, ExternalRawScore::MateMoves(-2));
        }
    }

    // Auxiliary raw projection only. Borrow original DTOs after original drain;
    // never reconstruct a view/payload, parse Debug, or issue acceptance.
    mod scalar_wire {
        use super::{DiagnosticError, HotEvent, opaque};
        use rz_contracts::pals::{RepresentationKey, Role, SearchAuthority};
        use rz_contracts::pals_adaptive::{
            CpuJobEnvelope, CpuJobId, ReviewViewSeal, RoleTicketEnvelope,
        };
        use rz_contracts::{Deadline, Digest, ExecutionId, RequestId, StateIdentity};
        use rz_search::pals::coordinator::{FrameId, StepExecution};
        use rz_search::pals::engine::adaptive_port::{
            AdaptivePublishedRepairLine, AdaptiveRepairLineInput, AdaptiveRoleConsumptionEvent,
            AdaptiveRoleConsumptionOutcome,
        };
        use rz_uci::engine::ProcessClock;
        use rz_uci::pals_adaptive::{
            AdaptiveCpuSliceDiagnosticError, AdaptiveCpuSliceDiagnosticSnapshot,
            AdaptiveCpuSliceObservation, AdaptiveCpuSliceResultObservation,
            AdaptivePendingQuestionRow, ContinuationDiagnosticError, ContinuationDiagnosticOptions,
            ContinuationDiagnosticSnapshot, ContinuationPendingObservation,
            ContinuationResolutionObservation, ContinuationResolutionSource,
            ContinuationTransitionKind,
        };
        use serde_json::{Value, json};
        use std::time::Instant;
        const DOMAIN: &str = "rovezero.continuation-diagnostic-scalar/3";
        fn hex(value: Digest) -> String {
            const HEX: &[u8; 16] = b"0123456789abcdef";
            let mut text = String::with_capacity(64);
            for byte in value.0 {
                text.push(HEX[(byte >> 4) as usize] as char);
                text.push(HEX[(byte & 15) as usize] as char);
            }
            text
        }
        fn authority(a: &SearchAuthority) -> Value {
            json!({"domain":"search_authority","process_epoch":a.epoch.0.to_string(),
                "game_generation":a.game.0.to_string(),"root_generation":a.root.0.to_string(),
                "implementation_digest":hex(a.implementation)})
        }
        fn request(id: RequestId) -> Value {
            json!({"domain":"role_request","process_epoch":id.epoch.0.to_string(),"request_sequence":id.sequence.to_string()})
        }
        fn execution(id: ExecutionId) -> Value {
            json!({"domain":"execution","process_epoch":id.epoch.0.to_string(),"execution_sequence":id.sequence.to_string()})
        }
        fn frame(id: FrameId) -> Value {
            json!({"domain":"refinement_frame","authority":authority(&id.authority),"frame_sequence":id.sequence.to_string()})
        }
        fn job(id: CpuJobId) -> Value {
            json!({"domain":"cpu_job","authority":authority(&id.authority),"job_sequence":id.sequence.to_string()})
        }
        fn state(id: StateIdentity) -> Value {
            json!({"domain":"rules_state","rules_state_owner":id.owner.0.to_string(),
                "state_revision":id.revision.0.to_string(),"semantic_digest":hex(id.semantic)})
        }
        fn deadline(value: Deadline) -> Value {
            json!({"domain":"monotonic_deadline","process_clock_epoch":value.clock.0.0.to_string(),
                "deadline_at_ns":value.at.0.to_string()})
        }
        fn original_instant(clock: &ProcessClock, instant: Instant) -> Value {
            match clock.tick_at(instant) {
                Ok(tick) => {
                    json!({"status":"COPIED_ORIGINAL_CLOCK","process_clock_epoch":clock.epoch().0.to_string(),
                    "monotonic_tick_ns":tick.0.to_string()})
                }
                Err(error) => {
                    json!({"status":"UNAVAILABLE","process_clock_epoch":clock.epoch().0.to_string(),
                    "monotonic_tick_ns":null,"original_clock_error_debug":opaque(&error)})
                }
            }
        }
        fn role(value: Role) -> &'static str {
            match value {
                Role::Proposer => "Proposer",
                Role::Critic => "Critic",
                Role::Validator => "Validator",
            }
        }
        fn view(value: &ReviewViewSeal) -> Value {
            json!({"domain":"review_view_seal","authority":authority(&value.authority),
                "view_sequence":value.view_sequence.to_string(),"context_policy_digest":hex(value.context_policy),
                "view_digest":hex(value.view_digest),"root_state":state(value.root_state),
                "actual_tokens":value.actual_tokens,"selected_records":value.selected_records,
                "anchors":value.anchors,"length_bucket":value.length_bucket})
        }
        fn representation(value: &RepresentationKey) -> Value {
            json!({"input_digest":hex(value.input),"history_digest":hex(value.history),
                "candidates_digest":hex(value.candidates),"mask_positions_digest":hex(value.mask_positions),
                "model_digest":hex(value.model),"encoding_digest":hex(value.encoding),
                "precision":opaque(&value.precision),"frozen_model_epoch":value.frozen_epoch.to_string(),
                "record_revision":value.record_revision.to_string(),"role":role(value.role)})
        }
        fn ticket(value: &RoleTicketEnvelope) -> Value {
            json!({"request":request(value.id),"authority":authority(&value.authority),
                "ticket_question_digest":hex(value.question),"view":view(&value.view),
                "representation":representation(&value.representation),"role":role(value.role),
                "original_deadline":deadline(value.original_deadline)})
        }
        fn step(value: StepExecution) -> Value {
            match value {
                StepExecution::Role(id) => json!({"kind":"Role","request":request(id)}),
                StepExecution::Cpu {
                    job: id,
                    execution: run,
                } => json!({"kind":"Cpu","job":job(id),"execution":execution(run)}),
                StepExecution::ExternalCpu {
                    job: id,
                    execution: run,
                } => json!({"kind":"ExternalCpu","job":job(id),"execution":execution(run)}),
                StepExecution::ProfiledExternal {
                    question,
                    execution: run,
                } => json!({"kind":"ProfiledExternal","question":{
                    "domain":"reciprocal_question","authority":authority(&question.authority),
                    "question_sequence":question.sequence.to_string()},"execution":execution(run)}),
            }
        }
        fn hot(value: &HotEvent) -> Value {
            match value {
                HotEvent::NativeSubmitted { ticket: value } => {
                    json!({"source":"NativeSubmitted","ticket":ticket(value),"native_os_owner":null})
                }
                HotEvent::NativeCompleted {
                    ticket: value,
                    physical,
                    output_present,
                    evidence,
                    invocation,
                } => json!({
                    "source":"NativeCompleted","ticket":ticket(value),"physical_completion":opaque(physical),
                    "output_present":output_present,"native_os_owner":null,
                    "execution":evidence.and_then(|e|e.execution).map(execution),
                    "completed_nn_inputs":evidence.and_then(|e|e.completed_nn_inputs),
                    "measurement_complete":evidence.map(|e|e.measurement_complete),
                    "physical_fresh":evidence.and_then(|e|e.physical_fresh),
                    "native_accounting":evidence.and_then(|e|e.accounting).map(|a|json!({
                        "domain":"ticket-native-owner-accounting/1",
                        "public_nn_runs_attempted":a.public_nn_runs_attempted(),
                        "public_nn_runs_completed":a.public_nn_runs_completed(),
                        "public_nn_runs_failed_known":a.public_nn_runs_failed_known(),
                        "role_nn_runs_attempted":a.role_nn_runs_attempted(),
                        "role_nn_runs_completed":a.role_nn_runs_completed(),
                        "role_nn_runs_failed_known":a.role_nn_runs_failed_known(),
                        "validator_nn_runs_attempted":a.validator_nn_runs_attempted(),
                        "validator_nn_runs_completed":a.validator_nn_runs_completed(),
                        "validator_nn_runs_failed_known":a.validator_nn_runs_failed_known(),
                        "scope":"ORIGINAL_TICKET_OWNER_DELTAS_NOT_LOGICAL_CONSUMPTION_OR_PROCESS_TOTALS"})),
                    "physical_invocation":invocation.map(|i|json!({"invocation_key":hex(i.invocation_key),
                        "accepted_seed_digest":i.accepted_seed_digest.map(hex),"physical_fresh":i.physical_fresh}))
                }),
                HotEvent::NativeRetired {
                    ticket: value,
                    accepted_by_native,
                } => json!({
                    "source":"NativeRetired","ticket":ticket(value),"accepted_by_native":accepted_by_native,
                    "semantic_consumption":null,"native_os_owner":null}),
                HotEvent::Dispatch {
                    ticket: value,
                    source_question,
                    source_view,
                    stage,
                } => json!({
                    "source":"SearchDispatch","ticket":ticket(value),"source_question_digest":hex(*source_question),
                    "submitted_view":view(source_view),"stage":opaque(stage)}),
                HotEvent::CpuSliceInstall(result) => match result {
                    Ok(()) => json!({"source":"CpuSliceInstall","install_return":"Ok"}),
                    Err(error) => {
                        json!({"source":"CpuSliceInstall","install_return":"Err","error_kind":opaque(error)})
                    }
                },
            }
        }
        fn pending_row(row: &AdaptivePendingQuestionRow) -> Value {
            json!({"preparation":{"domain":"native_preparation","preparation_sequence":row.preparation.0.to_string()},
                "frame":row.frame.map(frame),"stage":opaque(&row.stage),"source_question_digest":hex(row.question),
                "view":view(&row.view),"retained_view_digest":hex(row.retained_view_digest),
                "retained_view_seal_digest":hex(row.retained_view_seal_digest),
                "selected_record_revisions":row.selected_record_revisions.iter().map(u64::to_string).collect::<Vec<_>>(),
                "ticket":row.ticket.as_ref().map(ticket),"registry_observed_phase":row.registry_observed_phase.map(|p|opaque(&p)),
                "dispatch_pending":row.dispatch_pending,"completion_received":row.completion_received,
                "consumption_recorded":row.consumption_recorded})
        }
        fn pending(event: &ContinuationPendingObservation, clock: &ProcessClock) -> Value {
            let snapshot = match &event.snapshot {
                Ok(snapshot) => {
                    json!({"status":"COPIED_SOURCE_DTO","authority":authority(&snapshot.authority),
                    "phase":opaque(&snapshot.phase),"sampled_tick_ns":snapshot.sampled_at.0.to_string(),
                    "original_deadline":deadline(snapshot.original_deadline),
                    "questions":snapshot.questions.iter().map(pending_row).collect::<Vec<_>>()})
                }
                Err(error) => {
                    json!({"status":"UNAVAILABLE","source_error_kind":opaque(error),"questions":null})
                }
            };
            json!({"source":opaque(&event.source),"trigger_request":request(event.trigger),
                "queue_admitted_at":event.queue_admitted_at.map(|instant|original_instant(clock,instant)),
                "snapshot":snapshot})
        }

        fn published_repair_line(
            line: AdaptivePublishedRepairLine,
            source: &SearchAuthority,
        ) -> Value {
            json!({"domain":"search-published-repair-line/1","authority":authority(source),
                "line":{"domain":"search_store_line","store_local_index":line.line.0.to_string()},
                "observation":{"domain":"search_store_observation","store_local_index":line.observation.0.to_string()},
                "record_revision":line.record_revision.to_string(),
                "parent_record_revision":line.parent_revision.map(|v|v.to_string()),
                "supersedes_record_revision":line.supersedes_revision.map(|v|v.to_string()),
                "origin_state":{"domain":"search_store_state","store_local_index":line.origin_state.0.to_string()},
                "committed_plies":line.committed_plies,"first_move_uci":line.first_move.map(|m|m.to_string()),
                "source_boundary":"ORIGINAL_REPAIR_PUBLISH_RECORD_RETURNED_OK",
                "semantic_and_machine_acceptance":"REQUIRE_SEPARATE_ORIGINAL_EVENTS"})
        }
        fn repair_line_input(input: AdaptiveRepairLineInput) -> Value {
            let source = input.provenance;
            json!({"domain":"search-repair-line-input/1","consumer_refinement_stage":opaque(&input.refinement_stage),
                "source":{"producer_frame":frame(source.producer_frame),"producer_request":request(source.producer_request),
                    "submitted_source_question_digest":hex(source.submitted_question),"submitted_view_digest":hex(source.submitted_view),
                    "completed_evidence_digest":hex(source.completed_evidence),"original_deadline":deadline(source.original_deadline),
                    "published_repair_line":published_repair_line(source.published_line,&source.producer_frame.authority)},
                "submitted_model_proposal_digest":hex(input.model_proposal_sha256),
                "submitted_model_proposal_plies":input.model_proposal_plies,"submitted_repaired_plies":input.repaired_plies,
                "source_boundary":"ORIGINAL_SUBMITTED_OPAQUE_QUESTION",
                "consumption":"REQUIRES_SEPARATE_FIRST_CONSUMED_VIEW_EVENT"})
        }

        fn consumed(event: &AdaptiveRoleConsumptionEvent) -> Value {
            let outcome = match event.outcome {
                AdaptiveRoleConsumptionOutcome::SemanticResult => json!({"kind":"SemanticResult"}),
                AdaptiveRoleConsumptionOutcome::ConsumedView => json!({"kind":"ConsumedView"}),
                AdaptiveRoleConsumptionOutcome::DuplicateNoNewConsumption => {
                    json!({"kind":"DuplicateNoNewConsumption"})
                }
                AdaptiveRoleConsumptionOutcome::Rejected(reason) => {
                    json!({"kind":"Rejected","reason":opaque(&reason)})
                }
            };
            json!({"authority":authority(&event.authority),"request":request(event.request),"role":role(event.role),
                "ticket_question_digest":hex(event.question),"submitted_source_question_digest":event.submitted_question.map(hex),
                "ticket_view_digest":hex(event.ticket_view),"submitted_view_digest":hex(event.submitted_view),
                "frame":event.frame.map(frame),"source_stage":opaque(&event.stage),"outcome":outcome,
                "completed_evidence_digest":event.completed_evidence.map(hex),"prior_semantic_attempt":event.prior_semantic_attempt,
                "published_repair_line":event.published_repair_line.map(|line|published_repair_line(line,&event.authority)),
                "repair_line_input":event.repair_line_input.map(repair_line_input)})
        }
        fn resolved(event: &ContinuationResolutionObservation) -> Value {
            let transition = match event.transition {
                ContinuationTransitionKind::Continue { next_stage } => {
                    json!({"kind":"Continue","next_stage":opaque(&next_stage)})
                }
                ContinuationTransitionKind::Complete { scope } => {
                    json!({"kind":"Complete","scope":opaque(&scope)})
                }
                ContinuationTransitionKind::Unknown { reason } => {
                    json!({"kind":"Unknown","reason":reason})
                }
            };
            let source = match event.source {
                ContinuationResolutionSource::MachineTransition {
                    checked_rules_pair,
                    accepted,
                } => {
                    json!({"kind":"MachineTransition","checked_rules_pair":checked_rules_pair,"accepted":accepted})
                }
                ContinuationResolutionSource::ClosedFramePublication { published } => {
                    json!({"kind":"ClosedFramePublication","published":published})
                }
            };
            json!({"authority":authority(&event.authority),"evidence":{"frame":frame(event.evidence.frame),
                "stage":opaque(&event.evidence.stage),"execution":step(event.evidence.execution),
                "submitted_question_digest":hex(event.evidence.submitted_question),"submitted_view":view(&event.evidence.submitted_view),
                "completed_evidence_digest":hex(event.evidence.completed_evidence)},"transition":transition,"source":source,
                "checked_at":{"process_clock_epoch":event.clock.0.0.to_string(),"checked_tick_ns":event.checked_at.0.to_string()},
                "closed_frame":event.closed_frame.map(frame),"closed_scope":event.closed_scope.map(|s|opaque(&s)),
                "closed_unknown":event.closed_unknown})
        }
        fn job_envelope(value: &CpuJobEnvelope) -> Value {
            json!({"job":job(value.id),"state":state(value.state_identity),"question_digest":hex(value.question),
                "root_restriction_digest":hex(value.root_restriction),"evaluation_namespace_digest":hex(value.evaluation_namespace),
                "search_conditions_digest":hex(value.search_conditions),"total_native_node_credit":value.total_native_node_credit,
                "hard_deadline":deadline(value.hard_deadline)})
        }
        fn cpu_slice(value: &AdaptiveCpuSliceObservation) -> Value {
            let s = &value.scope;
            let g = &value.grant;
            let result = match &value.result {
                AdaptiveCpuSliceResultObservation::Report {
                    completion,
                    completed_depth,
                    reported_work,
                    score_scope,
                    reused_completed_depth,
                    yield_cause,
                    yield_ack_sequence,
                    publish_ack_sequence,
                    elapsed,
                } => json!({"kind":"Report", "completion":opaque(completion),
                        "completed_depth":completed_depth,"reported_work":{"nodes":reported_work.nodes,
                            "quiescence_nodes":reported_work.quiescence_nodes,"tt_hits":reported_work.tt_hits},
                        "score_scope":opaque(score_scope),"reused_completed_depth":reused_completed_depth,
                        "yield_cause":yield_cause.map(|c|opaque(&c)),"yield_ack_sequence":yield_ack_sequence.map(|s|s.to_string()),
                        "publish_ack_sequence":publish_ack_sequence.to_string(),"elapsed_duration_ns":elapsed.as_nanos().to_string()}),
                AdaptiveCpuSliceResultObservation::Error(error) => {
                    json!({"kind":"Error","source_error_kind":opaque(error)})
                }
            };
            json!({"source_sequence":value.sequence.to_string(),"scope":{"authority":authority(&s.authority),
                "frame":frame(s.frame),"stage":opaque(&s.stage),"question_digest":hex(s.question),"view":view(&s.view),
                "public_context_digest":hex(s.public_context),"job":job_envelope(&s.job),
                "requested_yield_sequence":s.requested_yield_sequence.map(|s|s.to_string()),"prior_yield_ack":s.prior_yield_ack.to_string(),
                "binding_progress_sequence":s.binding_progress_sequence.to_string(),"progress_since_slice":s.progress_since_slice,
                "feedback_already_forwarded":s.feedback_already_forwarded},
                "grant":{"job":job(g.job),"execution":execution(g.execution),
                    "cpu_owner":{"domain":"cpu_worker","worker_id":g.owner.0.to_string(),"os_owner":null},
                    "node_credit":g.node_credit,"soft_lease_duration_ns":g.soft_lease.as_nanos().to_string(),
                    "control_sequence":g.control_sequence.to_string()},
                "actual_work":value.actual_work.map(|w|json!({"nodes":w.nodes,"quiescence_nodes":w.quiescence_nodes,"tt_hits":w.tt_hits})),
                "result":result,"cumulative_nodes":value.cumulative_nodes,"report_progress_sequence":value.report_progress_sequence.to_string(),
                "continuation_retained":value.continuation_retained,"terminal":value.terminal,"admission_rejected":value.admission_rejected,
                "accounting_after_settlement":value.accounting_after_settlement.map(|a|json!({"global_spent":a.global_spent,
                    "global_reserved":a.global_reserved,"job_spent":a.job_spent,"job_reserved":a.job_reserved,"job_quarantined":a.job_quarantined}))})
        }
        fn presence(enabled: bool, empty: bool) -> &'static str {
            if !enabled {
                "DISABLED"
            } else if empty {
                "NOT_OBSERVED"
            } else {
                "SOURCE_RECORDS_PRESENT_ONLY"
            }
        }
        fn cpu_channel(
            value: &Option<
                Result<AdaptiveCpuSliceDiagnosticSnapshot, AdaptiveCpuSliceDiagnosticError>,
            >,
        ) -> Value {
            match value {
                None => json!({"enabled":false,"event_presence":"DISABLED","records":null}),
                Some(Err(error)) => {
                    json!({"enabled":true,"status":"UNAVAILABLE","source_error_kind":opaque(error),"records":null})
                }
                Some(Ok(s)) => {
                    json!({"enabled":true,"authority":s.authority.as_ref().map(authority),
                    "event_presence":presence(true,s.records.is_empty()),"records_max":s.records_max,
                    "source_unavailable":s.unavailable.map(|u|opaque(&u)),"records":s.records.iter().map(cpu_slice).collect::<Vec<_>>()})
                }
            }
        }
        fn continuation_channel(
            value: &Option<Result<ContinuationDiagnosticSnapshot, ContinuationDiagnosticError>>,
            options: Option<ContinuationDiagnosticOptions>,
            clock: &ProcessClock,
        ) -> Value {
            match value {
                None => json!({"enabled":false,"event_presence":"DISABLED","snapshot":null}),
                Some(Err(error)) => {
                    json!({"enabled":true,"status":"UNAVAILABLE","source_error_kind":opaque(error),"snapshot":null})
                }
                Some(Ok(s)) => {
                    let pending_enabled = options.is_some_and(|o| o.pending_snapshots_max != 0);
                    let resolution_enabled = options.is_some_and(|o| o.resolution_records_max != 0);
                    let role_enabled = options.is_some_and(|o| o.role_consumption);
                    json!({"enabled":true,"authority":s.authority.as_ref().map(authority),"source_unavailable_bits":s.unavailable_bits,
                        "pending":{"enabled":pending_enabled,"event_presence":presence(pending_enabled,s.pending.is_empty()),
                            "records":pending_enabled.then(||s.pending.iter().map(|e|pending(e,clock)).collect::<Vec<_>>())},
                        "resolutions":{"enabled":resolution_enabled,"event_presence":presence(resolution_enabled,s.resolutions.is_empty()),
                            "records":resolution_enabled.then(||s.resolutions.iter().map(resolved).collect::<Vec<_>>())},
                        "role_consumption":{"enabled":role_enabled,"event_presence":presence(role_enabled,s.role_events.is_empty()),
                            "install_observed":s.role_install_observed,"readback_original_root_fence":s.role_readback_root_fence,
                            "source_unavailable":s.role_unavailable.map(|u|opaque(&u)),
                            "records":role_enabled.then(||s.role_events.iter().map(consumed).collect::<Vec<_>>())}})
                }
            }
        }

        // Direct final DTO projection after the original finish_process call.
        // A logical return, root reuse, process shutdown and parent supervision
        // are separate facts. None of these observations replaces a guard.
        fn physical(value: rz_uci::search_driver::DriverPhysicalCompletion) -> &'static str {
            use rz_uci::search_driver::DriverPhysicalCompletion;
            match value {
                DriverPhysicalCompletion::Confirmed => "Confirmed",
                DriverPhysicalCompletion::Unknown => "Unknown",
            }
        }
        fn failure(value: &rz_uci::search_driver::SearchSessionFailure) -> Value {
            json!({"code":value.code,"detail":value.detail,
                "physical_completion_classification":physical(value.physical_completion),
                "classification_scope":"ORIGINAL_DRIVER_FAILURE_ONLY_NOT_A_PHYSICAL_FENCE"})
        }
        fn root_closure(value: &rz_uci::pals_adaptive::AdaptiveShutdownReceipt) -> Value {
            json!({"domain":"adaptive_root_shutdown_receipt","authority":authority(&value.authority),
                "native_owner_joined":value.native_owner_joined,
                "native_owner_quiesced":value.native_owner_quiesced,
                "native_physical_shutdown_confirmed":value.native_physical_shutdown_confirmed,
                "own_cpu_joined":value.own_cpu_joined,"expected_own_cpu_owners":value.expected_own_cpu_owners,
                "own_cpu_panicked":value.own_cpu_panicked,
                "external_shutdown_confirmed":value.external_shutdown_confirmed,
                "external_owner_quiesced":value.external_owner_quiesced,
                "auxiliary_shutdown_confirmed":value.auxiliary_shutdown_confirmed,
                "native_cpu_spent":value.native_cpu_spent,"native_cpu_reserved":value.native_cpu_reserved,
                "pending_tickets":value.pending_tickets,"quarantined":value.quarantined,
                "permits_semantic_closure":value.permits_semantic_closure(),
                "permits_root_reuse":value.permits_root_reuse(),
                "permits_archive_close":value.permits_archive_close(),
                "native_os_owner":null,"native_os_owner_observation":"NOT_OBSERVED_BY_THIS_CONSUMER"})
        }
        fn checker_shutdown(value: rz_search::cpu_checker::CheckerShutdown) -> Value {
            json!({"domain":"external_checker_shutdown_receipt",
                "process_identity":value.process_identity.map(|p|json!({"domain":"external_os_process",
                    "pid":p.pid.to_string(),"process_group":p.process_group.to_string(),"proc_start_ticks":p.proc_start_ticks.to_string()})),
                "stop_sent":value.stop_sent,"quit_sent":value.quit_sent,"exit_observed":value.exit_observed,
                "stdout_drained":value.stdout_drained,"stderr_drained":value.stderr_drained,
                "exit_code":value.exit_code,"exit_signal":value.exit_signal,"cleanup_complete":value.cleanup_complete,
                "quarantined":value.quarantined,"ownership_lost":value.ownership_lost,
                "stdout_bytes":value.stdout_bytes,"stderr_bytes":value.stderr_bytes})
        }
        fn semantic_completion(value: rz_search::pals::engine::PalsCompletion) -> &'static str {
            use rz_search::pals::engine::PalsCompletion;
            match value {
                PalsCompletion::RoundLimit => "RoundLimit",
                PalsCompletion::CpuNodeLimit => "CpuNodeLimit",
                PalsCompletion::RoleCallLimit => "RoleCallLimit",
                PalsCompletion::Deadline => "Deadline",
                PalsCompletion::Canceled => "Canceled",
                PalsCompletion::Capacity => "Capacity",
                PalsCompletion::Terminal => "Terminal",
            }
        }
        type OriginalDriverResult = Result<
            rz_uci::search_driver::SearchSessionReport,
            rz_uci::search_driver::SearchSessionFailure,
        >;
        type OriginalConsumerExecution = Result<OriginalDriverResult, DiagnosticError>;

        pub(super) fn runtime_closure(
            execution: &OriginalConsumerExecution,
            process: &Result<
                rz_uci::pals_adaptive::AdaptiveSessionProcessReceipt,
                rz_uci::search_driver::SearchSessionFailure,
            >,
            completion: &Result<
                rz_uci::pals_adaptive::AdaptiveCompletionDiagnostic,
                rz_contracts::ContractError,
            >,
        ) -> Value {
            let driver = match execution {
                Ok(Ok(_)) => json!({"status":"DRIVER_RETURNED_OK","failure":null,
                    "consumer_error":null,"physical_completion_classification":null}),
                Ok(Err(error)) => json!({"status":"DRIVER_RETURNED_ERROR","failure":failure(error),
                    "consumer_error":null,"physical_completion_classification":physical(error.physical_completion)}),
                Err(error) => {
                    json!({"status":"CONSUMER_FAILED_DRIVER_RETURN_NOT_OBSERVED","failure":null,
                    "consumer_error":error.to_string().chars().take(512).collect::<String>(),"physical_completion_classification":null})
                }
            };
            let process = match process {
                Ok(receipt) => json!({"status":"ORIGINAL_PROCESS_RECEIPT_RETURNED","failure":null,
                    "receipt":{"domain":"adaptive_session_process_receipt","process_epoch":receipt.epoch().0.to_string(),
                        "last_authority":receipt.last_authority().as_ref().map(authority),
                        "native_owner_joined":receipt.native_owner_joined(),
                        "native_physical_shutdown_confirmed":receipt.native_physical_shutdown_confirmed(),
                        "store_archive_closed":receipt.store_archive_closed(),
                        "permits_archive_close":receipt.permits_archive_close(),"permits_process_close":receipt.permits_process_close(),
                        "external_shutdown":receipt.external_shutdown().map(checker_shutdown),
                        "latest_root":receipt.latest_root().map(root_closure),
                        "native_os_owner":null,"native_os_owner_observation":"NOT_OBSERVED_BY_THIS_CONSUMER"}}),
                Err(error) => {
                    json!({"status":"ORIGINAL_PROCESS_FINISH_ERROR","failure":failure(error),
                    "receipt":null,"receipt_unavailable_reason":"ORIGINAL_FINISH_DID_NOT_RETURN_A_RECEIPT"})
                }
            };
            let completion = match completion {
                Ok(value) => {
                    json!({"status":if value.unavailable_reason.is_some(){"UNAVAILABLE"}else if value.last.is_some(){"OBSERVED"}else{"NOT_OBSERVED"},
                    "process_epoch":value.process_epoch.0.to_string(),"implementation_digest":hex(value.implementation),
                    "unavailable_reason":value.unavailable_reason,"last":value.last.as_ref().map(|last|json!({
                        "authority":authority(&last.authority),"semantic_completion":last.semantic_completion.map(semantic_completion),
                        "driver_return":match last.driver_return{rz_uci::pals_adaptive::AdaptiveCompletionReturn::Ok=>"Ok",rz_uci::pals_adaptive::AdaptiveCompletionReturn::Err=>"Err"},
                        "failure":last.failure.as_ref().map(|f|json!({"code":f.code,"detail":f.detail,"detail_truncated":f.detail_truncated,
                            "physical_completion_classification":physical(f.physical_completion)}))}))})
                }
                Err(error) => {
                    json!({"status":"UNAVAILABLE","last":null,"source_error_debug":opaque(error)})
                }
            };
            json!({"domain":"rovezero.continuation-diagnostic-runtime-closure/1",
                "driver_execution":driver,"final_semantic_observation":completion,"process_finish":process,
                "parent_supervision":"NOT_OBSERVED_BY_THIS_CONSUMER",
                "case_acceptance":"NOT_ISSUED_BY_DIAGNOSTIC_RUNNER"})
        }

        #[cfg(test)]
        mod tests {
            use super::*;
            use rz_contracts::{
                ClockDomain, GameGeneration, MonotonicTick, OwnerId, ProcessEpoch, RootGeneration,
                StateRevision,
            };
            use rz_search::pals::coordinator::{RefinementStage, StepEvidence};
            use rz_search::pals::engine::adaptive_port::AdaptiveRoleConsumptionStage;
            use rz_uci::pals_adaptive::{
                AdaptivePendingSnapshotUnavailable, ContinuationPendingSource,
            };
            use std::time::Duration;
            fn scope() -> SearchAuthority {
                SearchAuthority {
                    epoch: ProcessEpoch(42),
                    game: GameGeneration(5),
                    root: RootGeneration(7),
                    implementation: Digest([0xa5; 32]),
                }
            }
            fn seal() -> ReviewViewSeal {
                ReviewViewSeal {
                    authority: scope(),
                    view_sequence: 11,
                    context_policy: Digest([1; 32]),
                    view_digest: Digest([2; 32]),
                    root_state: StateIdentity {
                        owner: OwnerId(19),
                        revision: StateRevision(23),
                        semantic: Digest([3; 32]),
                    },
                    actual_tokens: 128,
                    selected_records: 0,
                    anchors: 1,
                    length_bucket: 128,
                }
            }

            #[test]
            fn closure_wire_keeps_failure_classification_separate_from_missing_receipt() {
                use rz_uci::search_driver::{DriverPhysicalCompletion, SearchSessionFailure};
                let execution = Ok(Err(SearchSessionFailure {
                    physical_completion: DriverPhysicalCompletion::Confirmed,
                    code: "OriginalDriverFailure",
                    detail: "authored DTO".into(),
                }));
                let process = Err(SearchSessionFailure::physical_completion_unknown());
                let completion = Ok(rz_uci::pals_adaptive::AdaptiveCompletionDiagnostic {
                    process_epoch: rz_contracts::ProcessEpoch(9),
                    implementation: Digest([4; 32]),
                    last: None,
                    observed_driver_returns: None,
                    successful_driver_return_reason_counts: None,
                    semantic_observed_but_driver_failed: None,
                    unobserved_completion_returns: None,
                    unavailable_reason: Some("authored unavailable"),
                });
                let value = runtime_closure(&execution, &process, &completion);
                assert_eq!(
                    value["driver_execution"]["physical_completion_classification"],
                    "Confirmed"
                );
                assert_eq!(
                    value["process_finish"]["failure"]["physical_completion_classification"],
                    "Unknown"
                );
                assert!(value["process_finish"]["receipt"].is_null());
                assert_eq!(value["final_semantic_observation"]["status"], "UNAVAILABLE");
                assert!(value["final_semantic_observation"]["last"].is_null());
                assert_eq!(value["parent_supervision"], "NOT_OBSERVED_BY_THIS_CONSUMER");
            }
            #[test]
            fn closure_wire_semantic_result_does_not_mint_driver_or_physical_success() {
                use rz_uci::pals_adaptive::{
                    AdaptiveCompletionDiagnostic, AdaptiveCompletionObservation,
                    AdaptiveCompletionReturn,
                };
                use rz_uci::search_driver::SearchSessionFailure;
                let execution = Err(Box::<dyn std::error::Error>::from(
                    "authored consumer failure",
                ));
                let process = Err(SearchSessionFailure::physical_completion_unknown());
                let completion = Ok(AdaptiveCompletionDiagnostic {
                    process_epoch: rz_contracts::ProcessEpoch(9),
                    implementation: Digest([4; 32]),
                    last: Some(AdaptiveCompletionObservation {
                        authority: scope(),
                        semantic_completion: Some(
                            rz_search::pals::engine::PalsCompletion::RoleCallLimit,
                        ),
                        driver_return: AdaptiveCompletionReturn::Err,
                        failure: None,
                    }),
                    observed_driver_returns: None,
                    successful_driver_return_reason_counts: None,
                    semantic_observed_but_driver_failed: None,
                    unobserved_completion_returns: None,
                    unavailable_reason: Some("authored prefix"),
                });
                let value = runtime_closure(&execution, &process, &completion);
                assert_eq!(
                    value["driver_execution"]["status"],
                    "CONSUMER_FAILED_DRIVER_RETURN_NOT_OBSERVED"
                );
                assert!(value["driver_execution"]["physical_completion_classification"].is_null());
                assert_eq!(
                    value["final_semantic_observation"]["last"]["semantic_completion"],
                    "RoleCallLimit"
                );
                assert_eq!(
                    value["final_semantic_observation"]["last"]["driver_return"],
                    "Err"
                );
                assert!(value["process_finish"]["receipt"].is_null());
            }

            #[test]
            fn repair_line_wire_preserves_publication_even_when_later_semantic_commit_failed() {
                use rz_search::pals::engine::adaptive_port::{
                    AdaptivePublishedRepairLine, AdaptiveRoleConsumptionRejection,
                };
                use rz_search::pals::store::{LineId, ObservationId, StateId};
                let event = AdaptiveRoleConsumptionEvent {
                    authority: scope(),
                    request: RequestId::new(scope().epoch, u64::MAX),
                    role: Role::Proposer,
                    question: Digest([3; 32]),
                    submitted_question: Some(Digest([3; 32])),
                    ticket_view: Digest([4; 32]),
                    submitted_view: Digest([4; 32]),
                    frame: Some(FrameId {
                        authority: scope(),
                        sequence: 77,
                    }),
                    stage: AdaptiveRoleConsumptionStage::SemanticCommit,
                    outcome: AdaptiveRoleConsumptionOutcome::Rejected(
                        AdaptiveRoleConsumptionRejection::Capacity,
                    ),
                    completed_evidence: None,
                    prior_semantic_attempt: Some(false),
                    repair_line_input: None,
                    published_repair_line: Some(AdaptivePublishedRepairLine {
                        line: LineId(11),
                        observation: ObservationId(12),
                        record_revision: u64::MAX,
                        parent_revision: Some(17),
                        supersedes_revision: None,
                        origin_state: StateId(2),
                        committed_plies: 3,
                        first_move: Some(rz_position::BoardMove::from_uci("e2e4").unwrap()),
                    }),
                };
                let value = consumed(&event);
                assert_eq!(value["outcome"]["kind"], "Rejected");
                assert!(value["completed_evidence_digest"].is_null());
                let line = &value["published_repair_line"];
                assert_eq!(line["record_revision"], u64::MAX.to_string());
                assert_eq!(line["line"]["domain"], "search_store_line");
                assert_eq!(line["origin_state"]["domain"], "search_store_state");
                assert_eq!(line["committed_plies"], 3);
                assert_eq!(line["first_move_uci"], "e2e4");
                let none = AdaptiveRoleConsumptionEvent {
                    published_repair_line: None,
                    repair_line_input: None,
                    ..event
                };
                assert!(consumed(&none)["published_repair_line"].is_null());
            }

            #[test]
            fn repair_input_wire_keeps_original_producer_and_consumer_domains_distinct() {
                use rz_search::pals::engine::adaptive_port::AdaptiveRepairLineProvenance;
                use rz_search::pals::store::{LineId, ObservationId, StateId};
                let source = AdaptiveRepairLineProvenance {
                    producer_frame: FrameId {
                        authority: scope(),
                        sequence: 5,
                    },
                    producer_request: RequestId::new(scope().epoch, 11),
                    submitted_question: Digest([11; 32]),
                    submitted_view: Digest([12; 32]),
                    completed_evidence: Digest([13; 32]),
                    original_deadline: Deadline {
                        clock: ClockDomain(scope().epoch),
                        at: MonotonicTick(1000),
                    },
                    published_line: AdaptivePublishedRepairLine {
                        line: LineId(7),
                        observation: ObservationId(8),
                        record_revision: 9,
                        parent_revision: Some(3),
                        supersedes_revision: None,
                        origin_state: StateId(1),
                        committed_plies: 4,
                        first_move: Some(rz_position::BoardMove::from_uci("e2e4").unwrap()),
                    },
                };
                let event = AdaptiveRoleConsumptionEvent {
                    authority: scope(),
                    request: RequestId::new(scope().epoch, 99),
                    role: Role::Critic,
                    question: Digest([21; 32]),
                    submitted_question: Some(Digest([21; 32])),
                    ticket_view: Digest([22; 32]),
                    submitted_view: Digest([22; 32]),
                    frame: Some(FrameId {
                        authority: scope(),
                        sequence: 6,
                    }),
                    stage: AdaptiveRoleConsumptionStage::SemanticCommit,
                    outcome: AdaptiveRoleConsumptionOutcome::SemanticResult,
                    completed_evidence: Some(Digest([23; 32])),
                    prior_semantic_attempt: Some(false),
                    published_repair_line: None,
                    repair_line_input: Some(AdaptiveRepairLineInput {
                        refinement_stage: RefinementStage::CriticRecheck,
                        provenance: source,
                        model_proposal_sha256: Digest([31; 32]),
                        model_proposal_plies: 4,
                        repaired_plies: Some(4),
                    }),
                };
                let value = consumed(&event);
                assert_eq!(value["request"]["request_sequence"], "99");
                assert_eq!(
                    value["repair_line_input"]["source"]["producer_request"]["request_sequence"],
                    "11"
                );
                assert_eq!(
                    value["repair_line_input"]["consumer_refinement_stage"],
                    "CriticRecheck"
                );
                assert_eq!(value["completed_evidence_digest"], hex(Digest([23; 32])));
                assert_eq!(
                    value["repair_line_input"]["source"]["completed_evidence_digest"],
                    hex(Digest([13; 32]))
                );
                assert!(value["published_repair_line"].is_null());
                assert_eq!(
                    value["repair_line_input"]["consumption"],
                    "REQUIRES_SEPARATE_FIRST_CONSUMED_VIEW_EVENT"
                );
                let missing = AdaptiveRoleConsumptionEvent {
                    repair_line_input: None,
                    ..event
                };
                assert!(consumed(&missing)["repair_line_input"].is_null());
            }

            #[test]
            fn scalar_wire_keeps_large_ids_and_typed_domains_distinct() {
                let a = scope();
                let r = request(RequestId::new(a.epoch, u64::MAX));
                let e = execution(ExecutionId {
                    epoch: a.epoch,
                    sequence: u64::MAX,
                });
                let f = frame(FrameId {
                    authority: a,
                    sequence: u64::MAX,
                });
                assert_eq!(r["request_sequence"], u64::MAX.to_string());
                assert_eq!(e["execution_sequence"], u64::MAX.to_string());
                assert_eq!(f["frame_sequence"], u64::MAX.to_string());
                assert_eq!(r["domain"], "role_request");
                assert_eq!(e["domain"], "execution");
                assert_eq!(f["domain"], "refinement_frame");
                assert_ne!(r, e);
                assert_eq!(f["authority"]["root_generation"], "7");
                assert_eq!(hex(Digest([0xa5; 32])), "a5".repeat(32));
                assert_eq!(view(&seal())["root_state"]["rules_state_owner"], "19");
            }
            #[test]
            fn scalar_wire_does_not_merge_semantic_consume_machine_and_publication() {
                let a = scope();
                let f = FrameId {
                    authority: a,
                    sequence: 29,
                };
                let request_id = RequestId::new(a.epoch, 31);
                let evidence = StepEvidence {
                    frame: f,
                    stage: RefinementStage::Repair,
                    execution: StepExecution::Role(request_id),
                    submitted_question: Digest([4; 32]),
                    submitted_view: seal(),
                    completed_evidence: Digest([5; 32]),
                };
                let accepted = ContinuationResolutionObservation {
                    authority: a,
                    evidence,
                    transition: ContinuationTransitionKind::Unknown {
                        reason: "projection-only source case",
                    },
                    checked_at: MonotonicTick(100),
                    clock: ClockDomain(a.epoch),
                    source: ContinuationResolutionSource::MachineTransition {
                        checked_rules_pair: false,
                        accepted: true,
                    },
                    closed_frame: Some(f),
                    closed_scope: None,
                    closed_unknown: Some("projection-only source case"),
                };
                let mut failed_publication = accepted;
                failed_publication.source =
                    ContinuationResolutionSource::ClosedFramePublication { published: false };
                let first = resolved(&accepted);
                let second = resolved(&failed_publication);
                assert_eq!(first["source"]["kind"], "MachineTransition");
                assert_eq!(first["source"]["accepted"], true);
                assert!(first["source"].get("published").is_none());
                assert_eq!(second["source"]["kind"], "ClosedFramePublication");
                assert_eq!(second["source"]["published"], false);
                assert!(second["source"].get("accepted").is_none());
                assert_eq!(first["evidence"], second["evidence"]);
                let semantic = AdaptiveRoleConsumptionEvent {
                    authority: a,
                    request: request_id,
                    role: Role::Proposer,
                    question: Digest([4; 32]),
                    submitted_question: Some(Digest([4; 32])),
                    ticket_view: Digest([2; 32]),
                    submitted_view: Digest([2; 32]),
                    frame: Some(f),
                    stage: AdaptiveRoleConsumptionStage::SemanticCommit,
                    outcome: AdaptiveRoleConsumptionOutcome::SemanticResult,
                    completed_evidence: Some(Digest([5; 32])),
                    prior_semantic_attempt: Some(false),
                    published_repair_line: None,
                    repair_line_input: None,
                };
                let mut first_consume = semantic;
                first_consume.stage = AdaptiveRoleConsumptionStage::ViewConsumption;
                first_consume.outcome = AdaptiveRoleConsumptionOutcome::ConsumedView;
                first_consume.completed_evidence = None;
                first_consume.prior_semantic_attempt = None;
                first_consume.frame = None;
                let semantic = consumed(&semantic);
                let consumed = consumed(&first_consume);
                assert_eq!(semantic["outcome"]["kind"], "SemanticResult");
                assert_eq!(consumed["outcome"]["kind"], "ConsumedView");
                assert!(consumed["completed_evidence_digest"].is_null());
                assert!(consumed["prior_semantic_attempt"].is_null());
                assert_eq!(semantic["request"], consumed["request"]);
            }
            #[test]
            fn scalar_wire_preserves_disabled_empty_failed_and_unknown_owner() {
                let clock = ProcessClock::new(scope().epoch);
                let empty = project(&clock, None, &Ok(Vec::new()), &None, &None, None);
                assert!(empty["native_os_owner"].is_null());
                assert!(empty["authority"].is_null());
                assert_eq!(empty["hot"]["event_presence"], "NOT_OBSERVED");
                assert_eq!(empty["cpu_slices"]["event_presence"], "DISABLED");
                assert!(empty["cpu_slices"]["records"].is_null());
                assert_eq!(empty["case_acceptance"], "NOT_ISSUED_BY_DIAGNOSTIC_RUNNER");
                let failed = project(
                    &clock,
                    None,
                    &Err("HotJournalPoisoned"),
                    &Some(Err(AdaptiveCpuSliceDiagnosticError::Poisoned)),
                    &Some(Err(ContinuationDiagnosticError::Busy)),
                    None,
                );
                assert_eq!(failed["hot"]["event_presence"], "UNAVAILABLE");
                assert!(failed["hot"]["records"].is_null());
                assert_eq!(failed["cpu_slices"]["status"], "UNAVAILABLE");
                assert!(failed["continuation"]["snapshot"].is_null());
            }
            #[test]
            fn scalar_wire_does_not_zero_fill_unavailable_snapshot_or_original_clock() {
                let before = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
                let clock = ProcessClock::new(scope().epoch);
                let event = ContinuationPendingObservation {
                    source: ContinuationPendingSource::QueueAdmitted,
                    trigger: RequestId::new(scope().epoch, 3),
                    queue_admitted_at: Some(before),
                    snapshot: Err(AdaptivePendingSnapshotUnavailable::RegistryBindingMismatch),
                };
                let wire = pending(&event, &clock);
                assert_eq!(wire["queue_admitted_at"]["status"], "UNAVAILABLE");
                assert!(wire["queue_admitted_at"]["monotonic_tick_ns"].is_null());
                assert_eq!(wire["snapshot"]["status"], "UNAVAILABLE");
                assert!(wire["snapshot"]["questions"].is_null());
                assert_eq!(wire["trigger_request"]["request_sequence"], "3");
            }
        }

        pub(super) fn project(
            clock: &ProcessClock,
            source_authority: Option<SearchAuthority>,
            hot_value: &Result<Vec<HotEvent>, &'static str>,
            cpu: &Option<
                Result<AdaptiveCpuSliceDiagnosticSnapshot, AdaptiveCpuSliceDiagnosticError>,
            >,
            continuation: &Option<
                Result<ContinuationDiagnosticSnapshot, ContinuationDiagnosticError>,
            >,
            options: Option<ContinuationDiagnosticOptions>,
        ) -> Value {
            json!({"domain":DOMAIN,"projection_scope":"TYPED_SOURCE_IDENTITIES_AND_STATUS_ONLY",
                "authority":source_authority.as_ref().map(authority),"original_process_clock_epoch":clock.epoch().0.to_string(),
                "native_os_owner":null,"native_os_owner_observation":"NOT_OBSERVED_BY_THIS_CONSUMER",
                "hot":{"event_presence":match hot_value {Ok(events)=>presence(true,events.is_empty()),Err(_)=>"UNAVAILABLE"},
                    "source_error":hot_value.as_ref().err(),"records":hot_value.as_ref().ok().map(|events|events.iter().map(hot).collect::<Vec<_>>())},
                "cpu_slices":cpu_channel(cpu),"continuation":continuation_channel(continuation,options,clock),
                "repair_to_critic_line_identity":{"status":"REQUEST_SCOPED_ORIGINAL_PROVENANCE_ONLY",
                    "event_field":"repair_line_input","acceptance":"ROOT_INDEPENDENT_AUDIT_REQUIRED",
                    "missing_input_provenance":"UNAVAILABLE_NOT_RECOVERED_BY_DIGEST_LENGTH_OR_FRAME"},
                "case_acceptance":"NOT_ISSUED_BY_DIAGNOSTIC_RUNNER"})
        }
    }

    pub fn main() -> ExitCode {
        let mut custom_input = None;
        match run(&mut custom_input) {
            Ok(true) => ExitCode::SUCCESS,
            Ok(false) => ExitCode::FAILURE,
            Err(error) => {
                let detail: String = error.to_string().chars().take(512).collect();
                let mut raw = json!({"domain":"rovezero.actual-continuation-diagnostic-raw/5",
                    "status":"PREPARATION_OR_PUBLICATION_FAILED","detail":detail,
                    "case_acceptance":"NOT_ISSUED_BY_DIAGNOSTIC_RUNNER",
                    "actual_nn_scope":"UNKNOWN_UNLESS_ORIGINAL_RAW_PROVES_IT","scalar_observations":null,"runtime_closure":null});
                record_custom_position(&mut raw, &custom_input);
                let _ = emit(&raw);
                ExitCode::FAILURE
            }
        }
    }
}
