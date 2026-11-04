//! Managed P-C V3 entry. Original failures and run owners remain in the
//! returned value until the caller explicitly ends its process lifetime.
//! Neither settlement nor this observation establishes GPU completion or
//! durable publication.
use super::ArenaError;
use super::managed_output_fd_v2::ManagedPlanLeaseRejectionV2;
use super::pc_cli_managed_v3::parse_managed_collection_cli_v3;
use super::pc_lock_v3::{ManagedOutputCloseEvidenceV3, PcCollectionRunV3};
use super::pc_managed_prepare_v3::{
    LongGamePreflightErrorV3, ManagedPcPrepareErrorV3, ManagedPcStartErrorV3,
    PreparedLongGamePreflightV3, prepare_long_game_preflight_v3, prepare_managed_pc_v3,
};
use crate::rovent_collect::pc_worker_v3::{
    PcDriverEvidenceErrorV3, PcDriverPollV3, PcSupervisorNoWorkerProofV3, SupervisorStartCustodyV3,
};
use std::any::Any;
use std::time::Instant;

fn bounded_detail(value: impl std::fmt::Display) -> String {
    value
        .to_string()
        .chars()
        .take(512)
        .map(|ch| if ch.is_control() { ' ' } else { ch })
        .collect()
}

fn prepare_error(error: &ManagedPcPrepareErrorV3) -> (&'static str, Option<String>) {
    match error {
        ManagedPcPrepareErrorV3::DescriptorUncertain(error) => {
            ("descriptor_uncertain", Some(bounded_detail(error)))
        }
        ManagedPcPrepareErrorV3::Observation(rejected) => match rejected {
            ManagedPlanLeaseRejectionV2::BeforeParts { error, .. } => {
                ("lease_before_parts", Some(bounded_detail(error)))
            }
            ManagedPlanLeaseRejectionV2::BeforeRequest { error, .. } => {
                ("lease_before_request", Some(bounded_detail(error)))
            }
            ManagedPlanLeaseRejectionV2::AfterRequest { error, .. } => {
                ("lease_after_request_uncertain", Some(bounded_detail(error)))
            }
        },
        ManagedPcPrepareErrorV3::Deadline { error, .. } => {
            ("deadline", Some(bounded_detail(format!("{error:?}"))))
        }
        ManagedPcPrepareErrorV3::Inputs { error, .. } => ("inputs", Some(bounded_detail(error))),
        ManagedPcPrepareErrorV3::Output { error, .. } => ("output", Some(bounded_detail(error))),
        ManagedPcPrepareErrorV3::WorkExpired { .. } => ("work_expired", None),
    }
}

fn start_error(error: &ManagedPcStartErrorV3) -> (&'static str, Option<String>) {
    match error {
        ManagedPcStartErrorV3::WorkExpired(_) => ("work_expired", None),
        ManagedPcStartErrorV3::Supervisor { error, .. } => {
            ("supervisor", Some(bounded_detail(error)))
        }
        ManagedPcStartErrorV3::SupervisorPanic { .. } => ("supervisor_panic_unknown", None),
        ManagedPcStartErrorV3::WorkExpiredAfterSupervisor { .. } => {
            ("work_expired_after_supervisor", None)
        }
        ManagedPcStartErrorV3::Launch { error, .. } => ("launch", Some(bounded_detail(error))),
        ManagedPcStartErrorV3::Panic { .. } => ("launch_panic_unknown", None),
    }
}

/// Progress is about the retained in-process settlement only. In particular,
/// `Settled` does not assert a successful native result or publication.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ManagedPcEntryProgressV3 {
    Pending,
    /// Only the selected read-only profile check completed.
    PreflightComplete,
    Settled,
    Failed,
    DeadlineReached,
}

enum EntryState {
    Selection(ArenaError),
    Bootstrap(ArenaError),
    Prepare(ManagedPcPrepareErrorV3),
    LongGamePreflight(Result<PreparedLongGamePreflightV3, LongGamePreflightErrorV3>),
    Start {
        original: ManagedPcStartErrorV3,
        recovered: Option<Box<PcCollectionRunV3>>,
        recovered_settled: bool,
        no_worker_proof: Option<PcSupervisorNoWorkerProofV3>,
        no_worker_released: bool,
        output_close: Option<ManagedOutputCloseEvidenceV3>,
        deadline_reached: bool,
        cleanup_until: Instant,
        recovery_error: Option<&'static str>,
        panic: Option<Box<dyn Any + Send>>,
    },
    Run {
        original: Box<PcCollectionRunV3>,
        cleanup_until: Instant,
        settled: bool,
        deadline_reached: bool,
        failure: Option<&'static str>,
        panic: Option<Box<dyn Any + Send>>,
    },
}

/// Retains the original error or run, not a serialized replacement of it.
/// This owner is process-local; a later process cannot recover it from JSON.
#[must_use]
pub struct ManagedPcEntryResultV3 {
    original: EntryState,
}

/// Call at the single-threaded child entry before another thread can touch the
/// inherited descriptors. None means no V3 selector was present.
pub fn try_run_managed_pc_entry_v3(
    arguments: Vec<String>,
    entry: Instant,
) -> Option<ManagedPcEntryResultV3> {
    let selection = match parse_managed_collection_cli_v3(arguments) {
        Ok(Some(selection)) => selection,
        Ok(None) => return None,
        Err(error) => {
            return Some(ManagedPcEntryResultV3 {
                original: EntryState::Selection(error),
            });
        }
    };
    let bootstrap_until = match selection.bootstrap_deadline(entry) {
        Ok(deadline) => deadline,
        Err(error) => {
            return Some(ManagedPcEntryResultV3 {
                original: EntryState::Bootstrap(error),
            });
        }
    };
    if selection.long_game_preflight.is_some() {
        return Some(ManagedPcEntryResultV3 {
            original: EntryState::LongGamePreflight(prepare_long_game_preflight_v3(
                selection,
                bootstrap_until,
            )),
        });
    }
    let prepared = match prepare_managed_pc_v3(selection, bootstrap_until) {
        Ok(prepared) => prepared,
        Err(original) => {
            return Some(ManagedPcEntryResultV3 {
                original: EntryState::Prepare(original),
            });
        }
    };
    let cleanup_until = prepared.cleanup_until;
    let original = match prepared.start() {
        Ok(run) => EntryState::Run {
            original: Box::new(run),
            cleanup_until,
            settled: false,
            deadline_reached: false,
            failure: None,
            panic: None,
        },
        Err(original) => EntryState::Start {
            original,
            recovered: None,
            recovered_settled: false,
            no_worker_proof: None,
            no_worker_released: false,
            output_close: None,
            deadline_reached: false,
            cleanup_until,
            recovery_error: None,
            panic: None,
        },
    };
    Some(ManagedPcEntryResultV3 { original })
}

impl ManagedPcEntryResultV3 {
    /// A single bounded poll. The caller owns pacing and may not extend the
    /// original cleanup ceiling; `poll_until` is clamped to that ceiling.
    pub fn try_progress(&mut self, poll_until: Instant) -> ManagedPcEntryProgressV3 {
        if let EntryState::LongGamePreflight(outcome) = &self.original {
            return if outcome.is_ok() {
                ManagedPcEntryProgressV3::PreflightComplete
            } else {
                ManagedPcEntryProgressV3::Failed
            };
        }
        if let EntryState::Start {
            original,
            recovered,
            recovered_settled,
            no_worker_proof,
            no_worker_released,
            output_close,
            deadline_reached,
            cleanup_until,
            recovery_error,
            panic,
        } = &mut self.original
        {
            if *recovered_settled {
                return ManagedPcEntryProgressV3::Failed;
            }
            if panic.is_some() {
                return ManagedPcEntryProgressV3::Failed;
            }
            if Instant::now() >= *cleanup_until {
                *deadline_reached = true;
                return ManagedPcEntryProgressV3::DeadlineReached;
            }
            let until = poll_until.min(*cleanup_until);
            if Instant::now() >= until {
                return ManagedPcEntryProgressV3::Pending;
            }
            // This failure is returned before launch is attempted. Retire the
            // exact registered supervisor only after its no-worker proof, then
            // close the same output owner without asserting publication.
            if let ManagedPcStartErrorV3::WorkExpiredAfterSupervisor { prepared, owners } = original
            {
                if no_worker_proof.is_none() {
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        SupervisorStartCustodyV3::try_seal_no_worker(
                            owners.driver_custody(),
                            owners.native(),
                        )
                    }));
                    match result {
                        Ok(Ok(PcDriverPollV3::Pending)) => {
                            return ManagedPcEntryProgressV3::Pending;
                        }
                        Ok(Ok(PcDriverPollV3::Ready(Some(proof)))) => {
                            *no_worker_proof = Some(proof)
                        }
                        Ok(Ok(PcDriverPollV3::Ready(None))) => {
                            *recovery_error = Some("original supervisor unstarted state unknown");
                            return ManagedPcEntryProgressV3::Failed;
                        }
                        Ok(Err(
                            PcDriverEvidenceErrorV3::Poisoned(error)
                            | PcDriverEvidenceErrorV3::Invalid(error),
                        )) => {
                            *recovery_error = Some(error);
                            return ManagedPcEntryProgressV3::Failed;
                        }
                        Err(payload) => {
                            *panic = Some(payload);
                            return ManagedPcEntryProgressV3::Failed;
                        }
                    }
                }
                if !no_worker_proof
                    .as_ref()
                    .expect("proof retained")
                    .matches_original(
                        owners.driver_custody(),
                        owners.supervisor(),
                        owners.native(),
                    )
                {
                    *recovery_error = Some("original no-worker proof differs");
                    return ManagedPcEntryProgressV3::Failed;
                }
                if !*no_worker_released {
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        SupervisorStartCustodyV3::try_release_no_worker(owners.driver_custody())
                    }));
                    match result {
                        Ok(Ok(PcDriverPollV3::Pending)) => {
                            return ManagedPcEntryProgressV3::Pending;
                        }
                        Ok(Ok(PcDriverPollV3::Ready(true))) => *no_worker_released = true,
                        Ok(Ok(PcDriverPollV3::Ready(false))) => {
                            *recovery_error = Some("original unstarted release not confirmed");
                            return ManagedPcEntryProgressV3::Failed;
                        }
                        Ok(Err(
                            PcDriverEvidenceErrorV3::Poisoned(error)
                            | PcDriverEvidenceErrorV3::Invalid(error),
                        )) => {
                            *recovery_error = Some(error);
                            return ManagedPcEntryProgressV3::Failed;
                        }
                        Err(payload) => {
                            *panic = Some(payload);
                            return ManagedPcEntryProgressV3::Failed;
                        }
                    }
                }
                if output_close.is_none() {
                    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        prepared.output.try_close()
                    })) {
                        Ok(value) => *output_close = value,
                        Err(payload) => {
                            *panic = Some(payload);
                            return ManagedPcEntryProgressV3::Failed;
                        }
                    }
                }
                return if output_close.is_some() {
                    ManagedPcEntryProgressV3::Failed
                } else {
                    ManagedPcEntryProgressV3::Pending
                };
            }
            if recovered.is_none() {
                let (prepared, owners) = match original {
                    ManagedPcStartErrorV3::WorkExpiredAfterSupervisor { prepared, owners }
                    | ManagedPcStartErrorV3::Launch {
                        prepared, owners, ..
                    }
                    | ManagedPcStartErrorV3::Panic {
                        prepared, owners, ..
                    } => (prepared, owners),
                    // No fully constructed supervisor owner exists in these
                    // variants; preserve the exact failure rather than invent
                    // an outcome or claim partial registration is absent.
                    _ => return ManagedPcEntryProgressV3::Failed,
                };
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    PcCollectionRunV3::recover_original_poll(owners, &prepared.output)
                })) {
                    Ok(Ok(PcDriverPollV3::Ready(Some(run)))) => *recovered = Some(Box::new(run)),
                    // No outcome does not prove no worker; keep the exact
                    // original owner and retry within the original ceiling.
                    Ok(Ok(PcDriverPollV3::Pending | PcDriverPollV3::Ready(None))) => {
                        return ManagedPcEntryProgressV3::Pending;
                    }
                    Ok(Err(
                        PcDriverEvidenceErrorV3::Poisoned(error)
                        | PcDriverEvidenceErrorV3::Invalid(error),
                    )) => {
                        if recovery_error.is_none() {
                            *recovery_error = Some(error);
                        }
                        return ManagedPcEntryProgressV3::Failed;
                    }
                    Err(payload) => {
                        *panic = Some(payload);
                        return ManagedPcEntryProgressV3::Failed;
                    }
                }
            }
            let original_run = recovered.as_mut().expect("checked original run");
            return match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                original_run.try_settle(until)
            })) {
                // A recovered failure stays a failure even after its owner is settled.
                Ok(Ok(true)) => {
                    *recovered_settled = true;
                    ManagedPcEntryProgressV3::Failed
                }
                Ok(Ok(false)) if original_run.before_native_recovered() => {
                    ManagedPcEntryProgressV3::Failed
                }
                Ok(Ok(false)) => ManagedPcEntryProgressV3::Pending,
                Ok(Err(error)) => {
                    *recovery_error = Some(error);
                    ManagedPcEntryProgressV3::Failed
                }
                Err(payload) => {
                    *panic = Some(payload);
                    ManagedPcEntryProgressV3::Failed
                }
            };
        }
        let EntryState::Run {
            original,
            cleanup_until,
            settled,
            deadline_reached,
            failure,
            panic,
        } = &mut self.original
        else {
            return ManagedPcEntryProgressV3::Failed;
        };
        if *settled {
            return ManagedPcEntryProgressV3::Settled;
        }
        if failure.is_some() || panic.is_some() {
            return ManagedPcEntryProgressV3::Failed;
        }
        if Instant::now() >= *cleanup_until {
            *deadline_reached = true;
            return ManagedPcEntryProgressV3::DeadlineReached;
        }
        let until = poll_until.min(*cleanup_until);
        if Instant::now() >= until {
            return ManagedPcEntryProgressV3::Pending;
        }
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| original.try_settle(until)))
        {
            Ok(Ok(true)) => {
                *settled = true;
                ManagedPcEntryProgressV3::Settled
            }
            Ok(Ok(false)) if original.before_native_recovered() => ManagedPcEntryProgressV3::Failed,
            Ok(Ok(false)) => ManagedPcEntryProgressV3::Pending,
            Ok(Err(error)) => {
                *failure = Some(error);
                ManagedPcEntryProgressV3::Failed
            }
            Err(payload) => {
                *panic = Some(payload);
                ManagedPcEntryProgressV3::Failed
            }
        }
    }

    pub fn cleanup_until(&self) -> Option<Instant> {
        match &self.original {
            EntryState::Run { cleanup_until, .. } | EntryState::Start { cleanup_until, .. } => {
                Some(*cleanup_until)
            }
            _ => None,
        }
    }

    /// Diagnostic only. Errors and panic payloads remain in the original
    /// owner; no raw credentials or source file contents are exposed.
    pub fn observation(&self) -> serde_json::Value {
        if let EntryState::LongGamePreflight(outcome) = &self.original {
            return match outcome {
                Ok(prepared) => {
                    let digest: String = prepared
                        .profile
                        .canonical_sha256()
                        .iter()
                        .map(|byte| format!("{byte:02x}"))
                        .collect();
                    serde_json::json!({
                        "schema":"rz-rovent-long-game-preflight-observation/1",
                        "status":"validated",
                        "raw_profile_pin":prepared.profile.original_pin(),
                        "canonical_profile_sha256":digest,
                        "profile_file_owner":"retained_during_process",
                        "native_started":false,
                        "long_game_execution":"not_started",
                        "capacity_applied":false,
                        "gpu_physical_completion":"not_observed",
                    })
                }
                Err(error) => {
                    let detail = match error {
                        LongGamePreflightErrorV3::Selection(error) => bounded_detail(error),
                        LongGamePreflightErrorV3::Setup(error) => {
                            let (stage, detail) = prepare_error(error);
                            format!("{stage}: {}", detail.unwrap_or_default())
                        }
                        LongGamePreflightErrorV3::Read { error, .. } => bounded_detail(error),
                    };
                    serde_json::json!({
                        "schema":"rz-rovent-long-game-preflight-observation/1",
                        "status":"failed",
                        "error":detail,
                        "native_started":false,
                        "long_game_execution":"not_started",
                        "capacity_applied":false,
                    })
                }
            };
        }
        let before_native_recovered = match &self.original {
            EntryState::Run { original, .. } => original.before_native_recovered(),
            EntryState::Start {
                recovered: Some(original),
                ..
            } => original.before_native_recovered(),
            _ => false,
        };
        let no_worker_close = match &self.original {
            EntryState::Start {
                output_close: Some(evidence),
                ..
            } => Some(format!("{:?}", evidence.status())),
            _ => None,
        };
        let (recovery_error, recovery_panic) = match &self.original {
            EntryState::Start {
                recovery_error,
                panic,
                ..
            } => (*recovery_error, panic.is_some()),
            _ => (None, false),
        };
        let (stage, settlement, native, first_error, cleanup_deadline_elapsed) =
            match &self.original {
                EntryState::Selection(error) => (
                    "selection_error",
                    "not_started",
                    "not_started",
                    Some(("selection", Some(bounded_detail(error)))),
                    false,
                ),
                EntryState::Bootstrap(error) => (
                    "bootstrap_error",
                    "not_started",
                    "not_started",
                    Some(("bootstrap", Some(bounded_detail(error)))),
                    false,
                ),
                EntryState::LongGamePreflight(_) => (
                    "long_game_preflight",
                    "not_started",
                    "not_started",
                    None,
                    false,
                ),
                EntryState::Prepare(error) => (
                    "prepare_error",
                    "not_started",
                    "not_started",
                    Some(prepare_error(error)),
                    false,
                ),
                EntryState::Start {
                    original,
                    recovered_settled,
                    deadline_reached,
                    ..
                } => (
                    "start_error",
                    if *recovered_settled {
                        "settled_owner_retained"
                    } else {
                        "unknown"
                    },
                    "unknown",
                    Some(start_error(original)),
                    *deadline_reached,
                ),
                EntryState::Run {
                    original,
                    settled,
                    deadline_reached,
                    failure,
                    panic,
                    ..
                } => {
                    let native = if *settled {
                        match original
                            .recovery_key()
                            .with_settled(|value| value.native.is_ok())
                        {
                            Ok(Some(true)) => "returned_ok",
                            Ok(Some(false)) => "returned_error",
                            _ => "unknown",
                        }
                    } else {
                        "unknown"
                    };
                    let settlement = if *settled {
                        "settled_owner_retained"
                    } else {
                        "unknown"
                    };
                    let stage = if panic.is_some() {
                        "settlement_panic"
                    } else if failure.is_some() {
                        "settlement_error"
                    } else {
                        "run_returned"
                    };
                    let first_error = if panic.is_some() {
                        Some(("settlement_panic", None))
                    } else {
                        failure.map(|error| ("settlement", Some(bounded_detail(error))))
                    };
                    (stage, settlement, native, first_error, *deadline_reached)
                }
            };
        serde_json::json!({
            "schema":"rz-rovent-managed-pc-entry-observation/1",
            "stage":stage,
            "native":native,
            "first_error":first_error.map(|(kind,detail)| serde_json::json!({"kind":kind,"detail":detail})),
            "recovery_error":recovery_error,
            "recovery_panic":recovery_panic,
            "cleanup_deadline_elapsed":cleanup_deadline_elapsed,
            "owner_settlement":settlement,
            "before_native_scope_recovered":before_native_recovered,
            "no_worker_output_close":no_worker_close,
            "teacher_completion":"unknown",
            "gpu_physical_completion":"unknown",
            "output_publication":"unknown",
            "original_owner_scope":"process_lifetime_only",
            "training_executed":false
        })
    }

    /// Publication and independent physical evidence are not yet available
    /// through this entry, so even settled custody is not a success result.
    pub fn exit_code(&self) -> u8 {
        if matches!(&self.original, EntryState::LongGamePreflight(Ok(_))) {
            0
        } else {
            1
        }
    }
}
