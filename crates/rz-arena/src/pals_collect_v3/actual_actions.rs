//! Explicit actual action collection. Construction performs no inference.
//! Preparation runs one bounded policy-selected source probe; its original
//! captures and full process cost remain separate from enumerated members.
use super::*;
use rz_contracts::ProcessEpoch;
use rz_experiments::PalsEngineModeV5;
use rz_search::pals::engine::PalsCompletion;
use rz_search::pals::engine::adaptive_port::validator_port::{
    AdaptiveValidatorAction, ValidatorActionOutcome,
};
use rz_uci::pals_adaptive::{
    AdaptiveCollectionDispatchObserver, AdaptiveCompletedRootObserver, AdaptiveManagerError,
    AdaptiveRootAdmissionObserver, AdaptiveSessionProcessReceipt, AdaptiveShutdownReceipt,
};
use rz_uci::pals_adaptive_bootstrap::{
    ActualAdaptiveBackendV5, AdaptiveBootstrapError, AdaptiveCollectionHooksV5,
    AdaptiveTeacherAdmissionV5, PreparedAdaptiveProcessV5, prepare_collection_actual,
    prepare_collection_actual_with_cancellation,
};
use rz_uci::pals_adaptive_startup::PinnedPalsStartupV5;
use rz_uci::search_driver::SearchSessionDriver;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const MAX_CONTEXT_PROCESSES: usize = 8;
const MAX_ISSUED_PROCESS_EPOCHS: usize = 8192 * MAX_CONTEXT_PROCESSES;

/// The caller shares its real process epoch allocator with selfplay. A saved
/// source epoch and every previous action process epoch are rejected on reuse.
pub type GenerationActionProcessEpochIssuerV3 =
    Arc<dyn Fn() -> Result<ProcessEpoch, ArenaError> + Send + Sync>;

#[derive(Clone, Copy, Debug)]
pub struct GenerationActualActionLimitsV3 {
    /// Startup, the actual root, and physical close share this original bound.
    pub process_wall: Duration,
    /// Reserved inside process_wall, never added after a failed turn.
    pub shutdown_reserve: Duration,
}
impl GenerationActualActionLimitsV3 {
    fn validate(self) -> Result<(), ArenaError> {
        check(
            !self.process_wall.is_zero()
                && !self.shutdown_reserve.is_zero()
                && self.shutdown_reserve < self.process_wall,
            "actual action process needs a finite wall bound and internal cleanup reserve",
        )
    }
}

/// Original capture copies for review/persistence. Neither this snapshot nor
/// its JSON receipts grants a target, native fence, or process-close authority.
#[derive(Clone)]
pub struct GenerationActualActionRawCaptureV3 {
    pub records: Vec<GenerationRecordV3>,
    pub evidence: BTreeMap<String, serde_json::Value>,
    /// Original semantic outcomes, copied before any consuming verifier.
    /// These raw observations grant no completion or target authority.
    pub semantic_outcomes: Vec<serde_json::Value>,
    pub capture_error: Option<String>,
}

/// A source probe really executes the original policy-selected action. Its
/// cost includes startup through physical process close, and no probe outcome
/// is returned as one of the registered action group's members.
pub struct GenerationActualActionProbeReceiptV3 {
    registration: Arc<FrozenActionRegistrationV3>,
    root: AdaptiveShutdownReceipt,
    completion: PalsCompletion,
    process: AdaptiveSessionProcessReceipt,
    wall_cost: Duration,
    capture: GenerationActualActionRawCaptureV3,
    outcome: GenerationActionOutcomeV3,
}
impl GenerationActualActionProbeReceiptV3 {
    pub fn registration(&self) -> &FrozenActionRegistrationV3 {
        &self.registration
    }
    pub fn root_receipt(&self) -> &AdaptiveShutdownReceipt {
        &self.root
    }
    pub fn completion(&self) -> PalsCompletion {
        self.completion
    }
    pub fn process_receipt(&self) -> &AdaptiveSessionProcessReceipt {
        &self.process
    }
    pub fn wall_cost(&self) -> Duration {
        self.wall_cost
    }
    pub fn capture(&self) -> &GenerationActualActionRawCaptureV3 {
        &self.capture
    }
    /// Auxiliary policy-selected outcome; its training wall cost is unknown
    /// and it is never returned to the registered action-group collector.
    pub fn outcome(&self) -> &GenerationActionOutcomeV3 {
        &self.outcome
    }
}

#[derive(Clone, Debug)]
pub struct GenerationActualActionStartFailureV3 {
    pub epoch: ProcessEpoch,
    pub stage: &'static str,
    pub detail: String,
    pub original_deadline: Instant,
    pub startup_deadline: Instant,
    /// Observed bootstrap-return cost, without a physical-close assertion.
    pub observed_wall_cost: Duration,
}

#[derive(Default)]
struct RootAudit {
    records: Vec<GenerationRecordV3>,
    semantic_outcomes: Vec<serde_json::Value>,
    evidence: BTreeMap<String, serde_json::Value>,
}

// Both registrations copy the actual dispatch seals and Rules anchor without
// retaining ReviewView/StoreReadPin. One belongs to the returning collector;
// the other remains with the factory and its original probe receipt.
struct CapturedQuestion {
    returned: FrozenActionRegistrationV3,
    retained: Arc<FrozenActionRegistrationV3>,
}
#[allow(
    clippy::large_enum_variant,
    reason = "One active root closure keeps its original probe or isolated action owner inline through physical close without another allocation"
)]
enum RootClosure {
    Probe {
        question: CapturedQuestion,
        receipt: AdaptiveShutdownReceipt,
        completion: PalsCompletion,
        outcome: GenerationActionOutcomeV3,
    },
    Member(GenerationIsolatedActionV3),
}
struct ProcessOwner {
    prepared: PreparedAdaptiveProcessV5,
    epoch: ProcessEpoch,
    started: Instant,
    original_deadline: Instant,
    process_receipt: Option<AdaptiveSessionProcessReceipt>,
    closed_at: Option<Instant>,
    capture: Arc<Mutex<GenerationGameCaptureV3>>,
    closure: Arc<Mutex<Option<RootClosure>>>,
    audit: Arc<Mutex<RootAudit>>,
}
struct FailedStartOwner {
    capture: Arc<Mutex<GenerationGameCaptureV3>>,
    closure: Arc<Mutex<Option<RootClosure>>>,
    audit: Arc<Mutex<RootAudit>>,
}

/// Copy only already published observations. No boolean in this raw receipt
/// can recreate CompletedValidatorAction or an enumerated member authority.
fn snapshot_semantic_outcome(semantic: &ValidatorActionOutcome) -> serde_json::Value {
    let parent = semantic.parent_view();
    serde_json::json!({
        "schema": "rz-pals-generation-raw-action-observation/1",
        "ticket": super::capture::generation_role_ticket_json_v3(semantic.source()),
        "action_index": semantic.id().action.index(),
        "selection_provenance": format!("{:?}", semantic.selection_provenance()),
        "frozen_context_sha256": hex(&semantic.frozen_context_digest().0),
        "input_sha256": hex(&semantic.input().0),
        "source_output_sha256": hex(&semantic.source_output().0),
        "source_public_context_sha256": hex(&semantic.source_public_context().0),
        "source_consumption_sha256": hex(&semantic.source_consumption().0),
        "parent_question_sha256": hex(&semantic.parent_question().0),
        "parent_input_sha256": hex(&semantic.parent_input().0),
        "parent_public_context_sha256": hex(&semantic.parent_public_context().0),
        "parent_view": {
            "epoch": parent.authority.epoch.0,
            "game": parent.authority.game.0,
            "root": parent.authority.root.0,
            "view_sequence": parent.view_sequence,
            "context_policy_sha256": hex(&parent.context_policy.0),
            "view_sha256": hex(&parent.view_digest.0),
            "actual_tokens": parent.actual_tokens,
            "selected_records": parent.selected_records,
            "anchors": parent.anchors,
            "length_bucket": parent.length_bucket,
        },
        "accepted_task_logits_f32": semantic.accepted_task_logits().iter()
            .map(|value| format!("{:08x}", value.to_bits())).collect::<Vec<_>>(),
        "original_first_move": semantic.original_first_move().bits(),
        "action_support_mask": semantic.action_support_mask(),
        "input_store_observation_sequence": semantic.input_observation_sequence(),
        "observed_action_wall_ns": semantic.wall_cost().as_nanos().to_string(),
        "witnesses": semantic.witnesses().iter().map(|witness| serde_json::json!({
            "kind": format!("{:?}", witness.kind()),
            "store_observation_sequence": witness.observation_sequence(),
            "store_generation": witness.store_generation(),
            "packed_line": witness.line().iter()
                .map(|mv| Move16::pack(*mv).ok().map(|packed| packed.bits()))
                .collect::<Vec<_>>(),
            "semantic_identity_sha256": hex(&witness.witness_identity().0),
            "store_commit_sha256": hex(&witness.raw_evidence().0),
        })).collect::<Vec<_>>(),
        "executions": semantic.executions().iter().map(|id| serde_json::json!({
            "epoch": id.epoch.0, "sequence": id.sequence,
        })).collect::<Vec<_>>(),
        "role_requests": semantic.role_requests().iter().map(|id| serde_json::json!({
            "epoch": id.epoch.0, "sequence": id.sequence,
        })).collect::<Vec<_>>(),
    })
}
struct ContextOwner {
    source: GenerationSavedActionRootV3,
    plan: ActionOutcomeCollectionPlanV3,
    original_deadline: Instant,
    registration: Option<Arc<FrozenActionRegistrationV3>>,
    next_member: usize,
    processes: Vec<ProcessOwner>,
    start_failures: Vec<GenerationActualActionStartFailureV3>,
    failed_start_owners: Vec<FailedStartOwner>,
    failed: Option<String>,
}

/// Concrete production factory. Its caller supplies an already content-verified
/// shared/P/C intake registration; no constructor flag can promote metadata.
/// Every probe/member owns a separately prepared native/CPU_T/archive process.
pub struct ActualGenerationActionOutcomeFactoryV3 {
    pinned: Arc<PinnedPalsStartupV5>,
    pins: FrozenPcRegistrationV3,
    producer: GenerationProducerAuthorityV3,
    epoch_issuer: GenerationActionProcessEpochIssuerV3,
    limits: GenerationActualActionLimitsV3,
    issued_epochs: BTreeSet<u64>,
    context: Option<ContextOwner>,
    probe: Option<GenerationActualActionProbeReceiptV3>,
    cancellation: Option<Arc<AtomicBool>>,
}
impl ActualGenerationActionOutcomeFactoryV3 {
    pub fn new(
        pinned: Arc<PinnedPalsStartupV5>,
        pins: FrozenPcRegistrationV3,
        producer: GenerationProducerAuthorityV3,
        epoch_issuer: GenerationActionProcessEpochIssuerV3,
        limits: GenerationActualActionLimitsV3,
    ) -> Result<Self, ArenaError> {
        limits.validate()?;
        check(
            pins.content_verified()
                && !pinned.metadata_preflight
                && pinned.config.endpoint.mode == PalsEngineModeV5::Collection,
            "actual action factory requires content-verified PC and actual Collection startup",
        )?;
        Ok(Self {
            pinned,
            pins,
            producer,
            epoch_issuer,
            limits,
            issued_epochs: BTreeSet::new(),
            context: None,
            probe: None,
            cancellation: None,
        })
    }

    /// Share the caller's latched admission/control flag with the existing
    /// per-root SearchControl. No extra polling worker or renewed deadline.
    pub fn with_cancellation(mut self, cancellation: Arc<AtomicBool>) -> Self {
        self.cancellation = Some(cancellation);
        self
    }

    fn check_cancellation(&self) -> Result<(), ArenaError> {
        check(
            !self
                .cancellation
                .as_ref()
                .is_some_and(|flag| flag.load(Ordering::Acquire)),
            "actual action admission canceled; original owners retained",
        )
    }

    pub fn probe_receipt(&self) -> Option<&GenerationActualActionProbeReceiptV3> {
        self.probe.as_ref()
    }
    /// The next context requires this receipt to be taken for bounded durable
    /// persistence; a later probe must never silently overwrite original data.
    pub fn take_probe_receipt(&mut self) -> Option<GenerationActualActionProbeReceiptV3> {
        self.probe.take()
    }
    pub fn start_failures(&self) -> &[GenerationActualActionStartFailureV3] {
        self.context
            .as_ref()
            .map_or(&[], |owner| owner.start_failures.as_slice())
    }
    pub fn retained_capture(
        &self,
        process: usize,
    ) -> Result<GenerationActualActionRawCaptureV3, ArenaError> {
        let owned = self
            .context
            .as_ref()
            .and_then(|owner| owner.processes.get(process))
            .ok_or_else(|| invalid("unknown retained actual action process"))?;
        Self::snapshot_capture(&owned.capture, &owned.closure, &owned.audit)
    }
    /// A late bootstrap failure still owns its original capture channels.
    /// Its returned raw data is diagnostic and physical closure stays Unknown.
    pub fn retained_start_failure_capture(
        &self,
        failure: usize,
    ) -> Result<GenerationActualActionRawCaptureV3, ArenaError> {
        let owned = self
            .context
            .as_ref()
            .and_then(|owner| owner.failed_start_owners.get(failure))
            .ok_or_else(|| invalid("unknown retained actual action startup failure"))?;
        Self::snapshot_capture(&owned.capture, &owned.closure, &owned.audit)
    }
    fn snapshot_capture(
        capture: &Arc<Mutex<GenerationGameCaptureV3>>,
        closure: &Arc<Mutex<Option<RootClosure>>>,
        audit: &Arc<Mutex<RootAudit>>,
    ) -> Result<GenerationActualActionRawCaptureV3, ArenaError> {
        let mut capture = capture
            .try_lock()
            .map_err(|_| invalid("actual action capture poisoned or busy"))?;
        let capture_error = capture.poll().err().map(|error| error.to_string());
        let audit = audit
            .try_lock()
            .map_err(|_| invalid("actual action audit poisoned or busy"))?;
        let mut records: BTreeMap<_, _> = audit
            .records
            .iter()
            .chain(capture.accepted_records())
            .map(|record| (record.input_sha256.clone(), record.clone()))
            .collect();
        let mut raw = GenerationActualActionRawCaptureV3 {
            records: vec![],
            evidence: capture.retained_evidence().clone(),
            semantic_outcomes: audit.semantic_outcomes.clone(),
            capture_error,
        };
        raw.evidence.extend(audit.evidence.clone());
        if let Some(RootClosure::Member(member)) = &*closure
            .try_lock()
            .map_err(|_| invalid("actual action closure poisoned or busy"))?
        {
            records.insert(member.record.input_sha256.clone(), member.record.clone());
            records.extend(
                member
                    .additional_records
                    .iter()
                    .cloned()
                    .map(|record| (record.input_sha256.clone(), record)),
            );
            raw.evidence.extend(member.evidence.clone());
        }
        raw.records = records.into_values().collect();
        Ok(raw)
    }

    fn check_source(
        owner: &ContextOwner,
        root: &GenerationSavedActionRootV3,
    ) -> Result<(), ArenaError> {
        check(
            owner.source.game_id() == root.game_id()
                && owner.source.opening_id() == root.opening_id()
                && owner.source.source_epoch() == root.source_epoch()
                && owner
                    .source
                    .position()
                    .snapshot()
                    .same_state(&root.position().snapshot()),
            "actual action changed its saved source root/history/attribution",
        )
    }

    fn close_owned(
        &mut self,
        until: Instant,
    ) -> Result<Vec<AdaptiveSessionProcessReceipt>, ArenaError> {
        let owner = self
            .context
            .as_mut()
            .ok_or_else(|| invalid("no actual action context to close"))?;
        let mut failures = vec![];
        let mut receipts = vec![];
        for process in &mut owner.processes {
            if process.process_receipt.is_none() {
                match process
                    .prepared
                    .finish_process(until.min(process.original_deadline))
                {
                    Ok(receipt)
                        if receipt.permits_process_close() && receipt.epoch() == process.epoch =>
                    {
                        process.process_receipt = Some(receipt);
                        process.closed_at = Some(Instant::now());
                    }
                    Ok(_) => failures.push(format!(
                        "epoch {} has no actual process-close fence",
                        process.epoch.0
                    )),
                    Err(error) => failures.push(format!(
                        "epoch {} physical close unknown: {error}",
                        process.epoch.0
                    )),
                }
            }
            if let Some(receipt) = &process.process_receipt {
                receipts.push(receipt.clone());
            }
            if let Err(error) = process
                .capture
                .try_lock()
                .map_err(|_| invalid("actual action capture poisoned or busy"))
                .and_then(|mut capture| capture.poll().map(|_| ()))
            {
                failures.push(format!(
                    "epoch {} retained capture: {error}",
                    process.epoch.0
                ));
            }
        }
        // Bootstrap attempts cleanup itself under its original startup bound:
        // Prepared::finish_process/Drop, StartupNativeGuard and rejected-source
        // cleanup keep native/CPU/archive assets in custody, quarantining any
        // unclosed ownership state rather than freeing its physical leases.
        // Its error exposes no typed caller-owned process-close receipt. Retain
        // exact stages and captures as Unknown; no subsequent epoch is retried.
        if !owner.start_failures.is_empty() {
            failures.push("failed startup physical closure is not attested to this factory".into());
        }
        if failures.is_empty() {
            Ok(receipts)
        } else {
            Err(invalid(failures.join("; ")))
        }
    }

    /// Attempt to close the same retained action processes without extending
    /// their original context or process deadlines. `None` means no context
    /// was created, not that a native process was observed to complete.
    pub(super) fn attempt_original_cleanup(
        &mut self,
        until: Instant,
    ) -> Result<Option<Vec<AdaptiveSessionProcessReceipt>>, ArenaError> {
        let Some(owner) = self.context.as_ref() else { return Ok(None); };
        let original_bound = until.min(owner.original_deadline);
        self.close_owned(original_bound).map(Some)
    }

    fn fail_closed<T>(&mut self, error: ArenaError, until: Instant) -> Result<T, ArenaError> {
        if let Some(owner) = &mut self.context {
            owner.failed = Some(error.to_string());
        }
        match self.close_owned(until) {
            Ok(_) => Err(error),
            Err(close) => Err(invalid(format!(
                "{error}; original-bound context cleanup: {close}"
            ))),
        }
    }

    fn run_process(
        &mut self,
        root: &GenerationSavedActionRootV3,
        action: Option<AdaptiveValidatorAction>,
        chronology: Arc<Mutex<GenerationChronologyV3>>,
        until: Instant,
    ) -> Result<usize, ArenaError> {
        self.check_cancellation()?;
        let owner = self
            .context
            .as_ref()
            .ok_or_else(|| invalid("actual action context absent"))?;
        Self::check_source(owner, root)?;
        check(
            owner.failed.is_none()
                && owner.processes.len() + owner.start_failures.len() < MAX_CONTEXT_PROCESSES,
            "actual action owner failed or process quota exhausted",
        )?;
        let started = Instant::now();
        let original_deadline = started
            .checked_add(self.limits.process_wall)
            .ok_or_else(|| invalid("actual action process deadline overflow"))?
            .min(until)
            .min(owner.original_deadline);
        let hard_deadline = original_deadline
            .checked_sub(self.limits.shutdown_reserve)
            .ok_or_else(|| invalid("actual action cleanup reserve overflow"))?;
        check(
            started < hard_deadline,
            "actual action window has no startup/root/cleanup budget",
        )?;
        let epoch = (self.epoch_issuer)()?;
        check(
            epoch.0 != 0
                && epoch.0 != root.source_epoch()
                && self.issued_epochs.len() < MAX_ISSUED_PROCESS_EPOCHS
                && !self.issued_epochs.contains(&epoch.0),
            "actual action requires a fresh distinct process epoch",
        )?;
        self.issued_epochs.insert(epoch.0);
        let plan = owner.plan.clone();
        let expected = owner.registration.clone();
        check(
            action.is_none() == expected.is_none(),
            "actual source probe and member registration phases differ",
        )?;
        let closed_chronology = Arc::clone(&chronology);
        let (mut capture, [native, unused_native], [dispatch, unused_dispatch]) =
            GenerationGameCaptureV3::new(
                root.game_id().into(),
                root.opening_id().into(),
                plan.frozen_pc_epoch,
                root.source_epoch(),
                chronology,
            )?;
        drop(unused_native);
        drop(unused_dispatch);
        capture.register_teacher_producer(&self.producer)?;
        capture.activate_root(root.position())?;
        let [teacher, unused_teacher] = capture.teacher_observers();
        drop(unused_teacher);
        let teacher_admission: AdaptiveTeacherAdmissionV5 = Arc::new(move |admitted| {
            teacher
                .register_source(admitted)
                .map_err(|error| AdaptiveBootstrapError {
                    stage: "ActionTeacherAdmission",
                    detail: error.to_string(),
                })
        });
        let teacher_observer = capture.teacher_observers()[0].observer();
        let capture = Arc::new(Mutex::new(capture));
        let question: Arc<Mutex<Option<CapturedQuestion>>> = Arc::new(Mutex::new(None));
        let closure: Arc<Mutex<Option<RootClosure>>> = Arc::new(Mutex::new(None));
        let audit = Arc::new(Mutex::new(RootAudit::default()));
        let dispatch_capture = Arc::clone(&capture);
        let dispatch_question = Arc::clone(&question);
        let dispatch_expected = expected.clone();
        let dispatch_plan = plan.clone();
        let dispatch_pins = self.pins.clone();
        let dispatch_observer: AdaptiveCollectionDispatchObserver =
            Arc::new(move |ticket, input, actual| {
                // Drain already published events before reserving the next bounded
                // original ticket. This never reconstructs native input evidence.
                dispatch_capture
                    .lock()
                    .map_err(|_| AdaptiveManagerError::Invalid("actual action capture poisoned"))?
                    .poll()
                    .map_err(|e| AdaptiveManagerError::Semantic(e.to_string()))?;
                if let Some(actual) = actual {
                    let mut frozen = dispatch_question.lock().map_err(|_| {
                        AdaptiveManagerError::Invalid("actual action question poisoned")
                    })?;
                    if frozen.is_some() {
                        return Err(AdaptiveManagerError::Invalid(
                            "isolated process submitted a second Validator question",
                        ));
                    }
                    let registered = FrozenActionRegistrationV3::from_actual_question(
                        &dispatch_plan,
                        actual,
                        &dispatch_pins,
                    )
                    .map_err(|e| AdaptiveManagerError::Semantic(e.to_string()))?;
                    if dispatch_expected.as_ref().is_some_and(|expected| {
                        registered.frozen_context_sha256() != expected.frozen_context_sha256()
                            || registered.action_support_mask() != expected.action_support_mask()
                            || registered.actions() != expected.actions()
                    }) {
                        return Err(AdaptiveManagerError::Invalid(
                            "isolated process changed its preregistered actual question",
                        ));
                    }
                    let returned = FrozenActionRegistrationV3::from_actual_question(
                        &dispatch_plan,
                        actual,
                        &dispatch_pins,
                    )
                    .map_err(|e| AdaptiveManagerError::Semantic(e.to_string()))?;
                    *frozen = Some(CapturedQuestion {
                        returned,
                        retained: Arc::new(registered),
                    });
                }
                dispatch(ticket, input, actual)
            });
        let root_admission: AdaptiveRootAdmissionObserver<ActualAdaptiveBackendV5> =
            Arc::new(move |manager| match action {
                Some(action) => manager.enable_exclusive_registered_action(action),
                None => manager.enable_exclusive_action_capture(),
            });
        let closed_capture = Arc::clone(&capture);
        let closed_question = Arc::clone(&question);
        let closed_result = Arc::clone(&closure);
        let closed_audit = Arc::clone(&audit);
        let root_closure: AdaptiveCompletedRootObserver<ActualAdaptiveBackendV5> =
            Arc::new(move |manager, completion| {
                let mut captured = closed_capture
                    .lock()
                    .map_err(|_| AdaptiveManagerError::Invalid("actual action capture poisoned"))?;
                captured
                    .poll()
                    .map_err(|e| AdaptiveManagerError::Semantic(e.to_string()))?;
                let mut output = closed_result
                    .lock()
                    .map_err(|_| AdaptiveManagerError::Invalid("actual action closure poisoned"))?;
                if output.is_some() {
                    return Err(AdaptiveManagerError::Invalid(
                        "actual action root closed twice",
                    ));
                }
                // Preserve original raw semantic evidence before the member
                // verifier consumes any outcome or capture publication.
                {
                    let mut audit = closed_audit.lock().map_err(|_| {
                        AdaptiveManagerError::Invalid("actual action audit poisoned")
                    })?;
                    audit.records = captured.accepted_records().cloned().collect();
                    audit.evidence = captured.retained_evidence().clone();
                    audit.semantic_outcomes = manager
                        .driver()
                        .action_outcomes()
                        .map(snapshot_semantic_outcome)
                        .collect();
                }
                let mut actual = closed_question.lock().map_err(|_| {
                    AdaptiveManagerError::Invalid("actual action question poisoned")
                })?;
                let registered = actual.as_ref().ok_or(AdaptiveManagerError::Invalid(
                    "actual root issued no Validator question",
                ))?;
                *output = Some(match &expected {
                    Some(registered) => RootClosure::Member(
                        GenerationIsolatedActionV3::from_actual_manager(
                            manager,
                            &mut captured,
                            registered,
                        )
                        .map_err(|e| AdaptiveManagerError::Semantic(e.to_string()))?,
                    ),
                    None => {
                        // This private proof requires the exclusive scope to
                        // have completed. Accepted V plus a reusable root alone
                        // does not attest that the selected action finished.
                        manager.exclusive_action_closure_proof()?;
                        let receipt = manager.shutdown_receipt()?;
                        if !receipt.permits_root_reuse()
                            || receipt.counters.nn_input_count_unknown != 0
                            || receipt.counters.completed_nn_inputs == 0
                            || manager.failures().next().is_some()
                        {
                            return Err(AdaptiveManagerError::Invalid(
                                "source probe lacks actual native run and root fence",
                            ));
                        }
                        let mut validator_seen = false;
                        for record in captured.accepted_records() {
                            registered
                                .retained
                                .verify_native_pins(record, captured.retained_evidence())
                                .map_err(|e| AdaptiveManagerError::Semantic(e.to_string()))?;
                            validator_seen |= record.input.role == GenerationRoleV3::Validator;
                        }
                        if !validator_seen {
                            return Err(AdaptiveManagerError::Invalid(
                                "source probe has no accepted original Validator input",
                            ));
                        }
                        let semantic = manager.driver_mut().take_action_outcome().ok_or(
                            AdaptiveManagerError::Invalid("source probe has no completed action"),
                        )?;
                        if manager.driver().action_outcomes().next().is_some() {
                            return Err(AdaptiveManagerError::Invalid(
                                "source probe executed more than one action",
                            ));
                        }
                        let completed = manager.complete_exclusive_action(semantic)?;
                        let source_ticket = super::capture::generation_role_ticket_json_v3(
                            completed.semantic().source(),
                        );
                        let mut originals = captured.retained_evidence().values().filter(|raw| {
                            raw.get("schema").and_then(|v| v.as_str())
                                == Some("rz-pals-generation-submission-evidence/1")
                                && raw.get("ticket") == Some(&source_ticket)
                        });
                        let input_sha = originals
                            .next()
                            .and_then(|raw| raw.get("input_sha256"))
                            .and_then(|v| v.as_str())
                            .map(str::to_owned)
                            .ok_or(AdaptiveManagerError::Invalid(
                                "source probe lacks original action input",
                            ))?;
                        if originals.next().is_some() {
                            return Err(AdaptiveManagerError::Invalid(
                                "source probe action input is ambiguous",
                            ));
                        }
                        let record = captured
                            .accepted_records()
                            .find(|record| record.input_sha256 == input_sha)
                            .cloned()
                            .ok_or(AdaptiveManagerError::Invalid(
                                "source probe action input was not accepted",
                            ))?;
                        // Ordinary capture deliberately gives no known pairwise
                        // training wall cost or enumerated-member authority.
                        let mut extra = captured.retained_evidence().clone();
                        let outcome = capture_completed_validator_action_v3(
                            &record,
                            completed,
                            &mut *closed_chronology.lock().map_err(|_| {
                                AdaptiveManagerError::Invalid("collection clock poisoned")
                            })?,
                            &mut extra,
                        )
                        .map_err(|e| AdaptiveManagerError::Semantic(e.to_string()))?;
                        closed_audit
                            .lock()
                            .map_err(|_| {
                                AdaptiveManagerError::Invalid("actual action audit poisoned")
                            })?
                            .evidence
                            .extend(extra);
                        RootClosure::Probe {
                            question: actual.take().expect("actual registration checked"),
                            receipt,
                            completion,
                            outcome,
                        }
                    }
                });
                Ok(())
            });
        let hooks = AdaptiveCollectionHooksV5 {
            native_observer: Box::new(native),
            dispatch_observer,
            teacher_observer,
            teacher_admission,
            root_admission,
            root_closure,
            native_intake: None,
        };
        let startup = if let Some(cancellation) = &self.cancellation {
            prepare_collection_actual_with_cancellation(
                Arc::clone(&self.pinned),
                epoch,
                hooks,
                hard_deadline,
                Arc::clone(cancellation),
            )
        } else {
            prepare_collection_actual(Arc::clone(&self.pinned), epoch, hooks, hard_deadline)
        };
        let prepared = match startup {
            Ok(prepared) => prepared,
            Err(error) => {
                let owner = self
                    .context
                    .as_mut()
                    .expect("context retained before startup");
                owner
                    .start_failures
                    .push(GenerationActualActionStartFailureV3 {
                        epoch,
                        stage: error.stage,
                        detail: error.detail.clone(),
                        original_deadline,
                        startup_deadline: hard_deadline,
                        observed_wall_cost: started.elapsed(),
                    });
                owner.failed_start_owners.push(FailedStartOwner {
                    capture,
                    closure,
                    audit,
                });
                return Err(invalid(format!("actual action startup unknown: {error}")));
            }
        };
        let processes = &mut self
            .context
            .as_mut()
            .expect("context retained through startup")
            .processes;
        let index = processes.len();
        processes.push(ProcessOwner {
            prepared,
            epoch,
            started,
            original_deadline,
            process_receipt: None,
            closed_at: None,
            capture,
            closure,
            audit,
        });
        let process = &processes[index];
        let mut context = process
            .prepared
            .context(hard_deadline, original_deadline)
            .map_err(|e| invalid(e.to_string()))?;
        link_action_cancellation(&mut context.control, self.cancellation.as_ref());
        let played = process
            .prepared
            .driver()
            .run(root.position(), &context, &mut |_| {});
        // A successful run retires its exact pending output; a failed run keeps
        // the active capability until finish_process cancels/drains its owners.
        if played.is_ok() {
            process
                .prepared
                .retire_context(&context)
                .map_err(|e| invalid(e.to_string()))?;
        }
        let played = played.map_err(|e| invalid(format!("actual isolated root: {e}")));
        let closed = self.close_owned(original_deadline);
        match (played, closed) {
            (Ok(_), Ok(_)) => Ok(index),
            (Err(error), Ok(_)) => Err(error),
            (Ok(_), Err(error)) => Err(error),
            (Err(error), Err(close)) => {
                Err(invalid(format!("{error}; physical process close: {close}")))
            }
        }
    }
}

impl GenerationActionOutcomeFactoryV3 for ActualGenerationActionOutcomeFactoryV3 {
    fn prepare_context(
        &mut self,
        root: &GenerationSavedActionRootV3,
        plan: &ActionOutcomeCollectionPlanV3,
        until: Instant,
    ) -> Result<FrozenActionRegistrationV3, ArenaError> {
        self.check_cancellation()?;
        plan.validate()?;
        check(
            self.context.is_none()
                && self.probe.is_none()
                && self.pins.content_verified()
                && Instant::now() < until,
            "previous actual context/probe data must be closed and persisted before preparation",
        )?;
        self.context = Some(ContextOwner {
            source: root.clone(),
            plan: plan.clone(),
            original_deadline: until,
            registration: None,
            next_member: 0,
            processes: vec![],
            start_failures: vec![],
            failed_start_owners: vec![],
            failed: None,
        });
        let prepared = (|| {
            let index = self.run_process(
                root,
                None,
                Arc::new(Mutex::new(GenerationChronologyV3::default())),
                until,
            )?;
            let raw = self.retained_capture(index)?;
            let owner = self
                .context
                .as_mut()
                .expect("source probe context retained");
            let process = &owner.processes[index];
            let mut closure = process
                .closure
                .lock()
                .map_err(|_| invalid("actual action closure poisoned"))?;
            let Some(RootClosure::Probe { question, .. }) = closure.as_ref() else {
                return Err(invalid("actual source probe has no original root closure"));
            };
            question
                .returned
                .verify_collection_source(plan, &self.producer, root.position())?;
            question
                .retained
                .verify_collection_source(plan, &self.producer, root.position())?;
            let process_receipt = process
                .process_receipt
                .clone()
                .ok_or_else(|| invalid("source probe process close unavailable"))?;
            let wall_cost = process
                .closed_at
                .ok_or_else(|| invalid("source probe process close time unavailable"))?
                .saturating_duration_since(process.started);
            // All fallible validation precedes transfer of the original closure.
            let Some(RootClosure::Probe {
                question,
                receipt,
                completion,
                outcome,
            }) = closure.take()
            else {
                return Err(invalid("actual source probe has no original root closure"));
            };
            owner.registration = Some(Arc::clone(&question.retained));
            self.probe = Some(GenerationActualActionProbeReceiptV3 {
                registration: question.retained,
                root: receipt,
                completion,
                process: process_receipt,
                wall_cost,
                capture: raw,
                outcome,
            });
            Ok(question.returned)
        })();
        match prepared {
            Ok(registered) => Ok(registered),
            Err(error) => self.fail_closed(error, until),
        }
    }

    fn execute_isolated(
        &mut self,
        root: &GenerationSavedActionRootV3,
        registration: &FrozenActionRegistrationV3,
        action: usize,
        chronology: Arc<Mutex<GenerationChronologyV3>>,
        until: Instant,
    ) -> Result<GenerationIsolatedActionV3, ArenaError> {
        let executed = (|| {
            self.check_cancellation()?;
            let owner = self
                .context
                .as_ref()
                .ok_or_else(|| invalid("actual action context absent"))?;
            Self::check_source(owner, root)?;
            let expected = owner
                .registration
                .as_ref()
                .ok_or_else(|| invalid("actual action context not preregistered"))?;
            registration.verify_collection_source(&owner.plan, &self.producer, root.position())?;
            check(
                expected.frozen_context_sha256() == registration.frozen_context_sha256()
                    && expected.action_support_mask() == registration.action_support_mask()
                    && expected.actions() == registration.actions()
                    && expected.actions().get(owner.next_member) == Some(&action),
                "actual action is reordered or differs from frozen registration",
            )?;
            let selected = AdaptiveValidatorAction::ALL
                .get(action)
                .copied()
                .ok_or_else(|| invalid("actual action index outside immutable order"))?;
            let index = self.run_process(root, Some(selected), chronology, until)?;
            let owner = self.context.as_mut().expect("actual member owner retained");
            let mut closure = owner.processes[index]
                .closure
                .lock()
                .map_err(|_| invalid("actual action closure poisoned"))?;
            let Some(RootClosure::Member(member)) = closure.as_ref() else {
                return Err(invalid(
                    "isolated actual process returned no private member",
                ));
            };
            check(
                member.outcome().action.index() == action,
                "actual member completed another registered action",
            )?;
            let Some(RootClosure::Member(member)) = closure.take() else {
                unreachable!("validated member variant retained under the same mutex")
            };
            owner.next_member += 1;
            Ok(member)
        })();
        match executed {
            Ok(member) => Ok(member),
            Err(error) => self.fail_closed(error, until),
        }
    }

    fn close_context(
        &mut self,
        until: Instant,
    ) -> Result<GenerationActualActionContextCloseV3, ArenaError> {
        let receipts = self.close_owned(until)?;
        let closed = GenerationActualActionContextCloseV3::from_actual_session_receipts(&receipts)?;
        let owner = self
            .context
            .as_ref()
            .expect("context retained through actual close");
        if let Some(failed) = &owner.failed {
            return Err(invalid(format!(
                "actual action context failed; original captures retained: {failed}"
            )));
        }
        let registered = owner
            .registration
            .as_ref()
            .ok_or_else(|| invalid("actual context has no registered source question"))?;
        check(
            owner.next_member == registered.actions().len()
                && receipts.len() == owner.next_member + 1,
            "actual context closed before every registered member and source probe",
        )?;
        self.context = None;
        Ok(closed)
    }
}

fn link_action_cancellation(
    control: &mut rz_search::driver::SearchControl,
    cancellation: Option<&Arc<AtomicBool>>,
) {
    if let Some(cancellation) = cancellation {
        control.cancellation = Arc::clone(cancellation);
    }
}

impl Drop for ActualGenerationActionOutcomeFactoryV3 {
    fn drop(&mut self) {
        if let Some(owner) = &self.context {
            let until = owner.original_deadline;
            // Drop is only a final attempt at the original bound. Successful
            // receipts are kept; failed native owners retain their quarantine.
            let _ = self.close_owned(until);
        }
    }
}

#[cfg(test)]
mod cancellation_tests {
    use super::*;
    #[test]
    fn original_latched_control_reaches_current_and_future_action_roots_without_new_budget() {
        let deadline = Instant::now() + Duration::from_secs(1);
        let cancellation = Arc::new(AtomicBool::new(false));
        let mut first = rz_search::driver::SearchControl::new(deadline, 65_536);
        link_action_cancellation(&mut first, Some(&cancellation));
        cancellation.store(true, Ordering::Release);
        assert!(first.cancellation.load(Ordering::Acquire));
        let mut next = rz_search::driver::SearchControl::new(deadline, 65_536);
        link_action_cancellation(&mut next, Some(&cancellation));
        assert!(next.cancellation.load(Ordering::Acquire));
        assert!(Arc::ptr_eq(&first.cancellation, &next.cancellation));
        assert_eq!(first.deadline, deadline);
        assert_eq!(next.deadline, deadline);
        assert_eq!(next.max_simulations, 65_536);
    }
}
