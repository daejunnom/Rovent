//! Own NNUEC input-lock /2 launch custody. A returned run is an observation, not a
//! published corpus or a claim of native/teacher physical completion.
use super::*;
use crate::rovent_collect::own_hooks_lock_v2::{
    RoventNnuecCollectionBridgeV2, RoventNnuecCollectionWorkerExitV2, RoventNnuecCollectionWorkerV2,
};
use crate::rovent_collect::teacher::v2_session::RoventTeacherRosterEntryV2;
use crate::rovent_collect::teacher::lock_v2_policy::CommitTeacherPolicyFailureV2;
use crate::rovent_collect::teacher::lock_v2_row_pump::PumpItemV2;
use rz_experiments::RoventInputLockV2;
use rz_experiments::rovent_corpus_lock_v2::record::RoventCorpusManifestLockV2;
use rz_uci::rovent_bootstrap::launch::{
    RoventOwnNnuecCollectionLaunchFailureV2, run_own_nnuec_collection_v2_with_resource_admission,
};
use rz_uci::rovent_bootstrap::nnuec_session::{
    RoventNnuecCollectionHandoffV2, RoventNnuecHandoffTransferErrorV2,
    RoventNnuecTransferredHandoffV2,
};
use rz_uci::rovent_bootstrap::nnuec_session::handoff_escrow_v2::{
    HandoffEscrowErrorV2, HandoffEscrowKeyV2, HandoffEscrowLeaseV2,
};
use std::sync::atomic::Ordering;
use std::thread::JoinHandle;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OwnPumpCommitProgressV2 {
    Pending,
    Empty,
    NotPrepared,
    AlreadyAttached,
    Attached,
}

/// Keeps the original resource, native failure and teacher worker together.
/// Neither the live readback nor the thread exit establishes publication.
pub struct RoventNnuecCollectionRunV2 {
    resource: Arc<Mutex<resources::CollectionResourceCustody>>,
    original_lock: Arc<Mutex<Option<Arc<RoventInputLockV2>>>>,
    owner: RoventCollectionOwnerLockV2,
    bridge: Arc<RoventNnuecCollectionBridgeV2>,
    worker: Arc<Mutex<RoventNnuecCollectionWorkerV2>>,
    worker_thread: Option<JoinHandle<RoventNnuecCollectionWorkerExitV2>>,
    worker_not_started: bool,
    worker_exit: Option<RoventNnuecCollectionWorkerExitV2>,
    worker_thread_panicked: bool,
    worker_custody_poisoned: bool,
    transferred_handoff: Option<RoventNnuecTransferredHandoffV2>,
    recovered_handoff: Option<HandoffEscrowLeaseV2>,
    escrow_recovery_error: Option<ArenaError>,
    handoff_transfer_error: Option<RoventNnuecHandoffTransferErrorV2>,
    commit_failure: Option<Box<CommitTeacherPolicyFailureV2>>,
    native_failure: Option<RoventOwnNnuecCollectionLaunchFailureV2>,
    native_failure_taken: bool,
    pre_admission_failure_transferred: bool,
    resource_error: Option<String>,
}

impl RoventNnuecCollectionRunV2 {
    /// Advance a caller-owned assembly with this run's original root gate and
    /// handoff. The assembly keeps the original row on busy/error; success is
    /// only in-memory assembly, not corpus publication or teacher retirement.
    pub fn try_advance_assembly(&self, assembly: &mut RoventCorpusAssemblyLockV2) -> Result<bool, ArenaError> {
        if self.worker_thread.is_some() || self.worker_exit.is_none()
            || self.worker_thread_panicked || self.worker_custody_poisoned
            || matches!(self.worker_exit.as_ref(), Some(RoventNnuecCollectionWorkerExitV2::Panicked(_)))
        { return Ok(false); }
        let worker = match self.worker.try_lock() {
            Ok(worker) => worker,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(false),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(invalid("Collection /2 worker poisoned")),
        };
        let Some(handoff) = worker.original_handoff() else { return Ok(false); };
        let mut handoff = match handoff.try_lock() {
            Ok(handoff) => handoff,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(false),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(invalid("Collection /2 handoff poisoned")),
        };
        let admission = handoff.parts_mut().0.clone();
        let Some(gate) = self.bridge.try_begin_commit(&admission)? else { return Ok(false); };
        assembly.try_advance_fenced_own(&gate, &mut handoff)
    }
    /// Recover only the original installed admission after the Arena worker
    /// has stopped. The lease preserves custody and proves no completion.
    pub fn try_recover_original_handoff(&mut self) -> Result<bool, ArenaError> {
        if self.recovered_handoff.is_some() { return Ok(true); }
        if self.worker_thread.is_some() || self.worker_custody_poisoned
            || (self.worker_exit.is_none() && !self.worker_not_started && !self.worker_thread_panicked)
        {
            return Ok(false);
        }
        let Some(admission) = self.bridge.installed_admission_for_recovery() else { return Ok(false); };
        let lease = match HandoffEscrowKeyV2::try_recover_for_admission(admission) {
            Ok(Some(lease)) => lease,
            Ok(None) | Err(HandoffEscrowErrorV2::Busy | HandoffEscrowErrorV2::AlreadyLeased) => return Ok(false),
            Err(error) => return Err(invalid(format!("Collection /2 escrow recovery: {error:?}"))),
        };
        let delivery_observed = match lease.original().try_lock() {
            Ok(mut handoff) => {
                if !handoff.parts_mut().0.same_original_admission(admission) {
                    return Err(invalid("Collection /2 recovered handoff admission mismatch"));
                }
                handoff.handoff_delivery().is_some()
            }
            Err(std::sync::TryLockError::WouldBlock) => return Ok(false),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(invalid("Collection /2 recovered handoff poisoned")),
        };
        if !delivery_observed { return Ok(false); }
        self.recovered_handoff = Some(lease);
        Ok(true)
    }

    /// Transfer the original recovery error for a caller to retain. It is a
    /// collection observation, not a native or teacher physical failure.
    pub fn take_escrow_recovery_error(&mut self) -> Option<ArenaError> {
        self.escrow_recovery_error.take()
    }

    /// Transfer the original UCI failure and resource observation only for a
    /// proven no-thread, no-storage, pre-resource-admission failure. The
    /// caller must retain the original failure (including hook and observer)
    /// and the observation. This is not a successful run or a receipt.
    pub fn try_take_pre_admission_failure(
        &mut self,
    ) -> Result<Option<(RoventOwnNnuecCollectionLaunchFailureV2, serde_json::Value)>, ArenaError> {
        if self.pre_admission_failure_transferred
            || self.native_failure_taken
            || !self.worker_not_started
            || self.worker_custody_poisoned
            || self.worker_thread.is_some()
            || self.worker_thread_panicked
            || self.resource_error.is_some()
        {
            return Ok(None);
        }
        let Some(failure) = self.native_failure.as_ref() else { return Ok(None); };
        if failure.physical_completion_unknown()
            || failure.parsed_storage_selected() != Some(false)
            || !failure.settled_observer_available()
            || failure.observer_custody_retained()
        {
            return Ok(None);
        }
        let resource = match self.resource.try_lock() {
            Ok(resource) => resource,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(invalid("Collection /2 original resource custody poisoned")),
        };
        if !resource.no_admission_started() {
            return Ok(None);
        }
        let observation = resource.observation();
        drop(resource);
        let original = self.native_failure.take().expect("checked original failure");
        self.native_failure_taken = true;
        self.pre_admission_failure_transferred = true;
        Ok(Some((original, observation)))
    }
    /// Attempt one in-memory policy attachment after the original worker has
    /// joined. Pending/other items stay in the original worker. This operation
    /// does not publish rows or assert teacher/process physical completion.
    /// Boundary errors are transferred to the caller; the original row and
    /// prepared evidence remain in the worker. Policy-commit failures are
    /// instead retained by this run, including any panic payload.
    pub fn try_commit_pump_head(&mut self) -> Result<OwnPumpCommitProgressV2, ArenaError> {
        if matches!(self.worker_exit.as_ref(), Some(RoventNnuecCollectionWorkerExitV2::Panicked(_))) {
            return Err(invalid("original row worker panicked; exit payload remains retained"));
        }
        if self.worker_thread_panicked || self.worker_custody_poisoned {
            return Err(invalid("original row worker completion is unknown"));
        }
        if self.worker_thread.is_some() || self.worker_exit.is_none() {
            return Ok(OwnPumpCommitProgressV2::Pending);
        }
        if self.commit_failure.is_some() {
            return Err(invalid("original row commit failure remains retained"));
        }
        let mut worker = match self.worker.try_lock() {
            Ok(worker) => worker,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(OwnPumpCommitProgressV2::Pending),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(invalid("original row worker poisoned")),
        };
        let Some(handoff) = worker.original_handoff() else { return Ok(OwnPumpCommitProgressV2::Pending); };
        let mut handoff = match handoff.try_lock() {
            Ok(handoff) => handoff,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(OwnPumpCommitProgressV2::Pending),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(invalid("original row handoff poisoned")),
        };
        let admission = handoff.parts_mut().0.clone();
        let Some(gate) = self.bridge.try_begin_commit(&admission)? else { return Ok(OwnPumpCommitProgressV2::Pending); };
        let Some(item) = worker.first_pump_item_mut() else { return Ok(OwnPumpCommitProgressV2::Empty); };
        let PumpItemV2::Prepared { row, evidence, sealed } = item else {
            return Ok(OwnPumpCommitProgressV2::NotPrepared);
        };
        if sealed.is_some() { return Ok(OwnPumpCommitProgressV2::AlreadyAttached); }
        let result = self.owner.try_with_fenced_own_producer_mut(row, &gate, &mut handoff,
            |producer, row| producer.attach_prepared_teacher_policy_v2(row, evidence))?;
        match result {
            None => Ok(OwnPumpCommitProgressV2::Pending),
            Some(Ok(original_sealed)) => { *sealed = Some(original_sealed); Ok(OwnPumpCommitProgressV2::Attached) }
            Some(Err(original)) => {
                self.commit_failure = Some(original);
                Err(invalid("original row policy commit failed and is retained"))
            }
        }
    }
    /// Recover the session's original handoff only after joining this run's
    /// original bridge worker. This is custody, not publication or a teacher
    /// semantic-success receipt. The same run keeps the worker and rows.
    pub fn try_transfer_joined_handoff(
        &mut self,
    ) -> Result<bool, RoventNnuecHandoffTransferErrorV2> {
        if self.transferred_handoff.is_some() {
            return Ok(true);
        }
        if self.worker_thread.is_some() || self.worker_exit.is_none() || self.worker_thread_panicked || self.worker_custody_poisoned {
            return Ok(false);
        }
        let handoff = {
            let worker = self.worker.try_lock().map_err(|error| match error {
                std::sync::TryLockError::WouldBlock => RoventNnuecHandoffTransferErrorV2::Busy,
                std::sync::TryLockError::Poisoned(_) => RoventNnuecHandoffTransferErrorV2::Poisoned,
            })?;
            let Some(handoff) = worker.original_handoff() else { return Ok(false); };
            handoff
        };
        let admission = {
            let mut original = handoff.try_lock().map_err(|error| match error {
                std::sync::TryLockError::WouldBlock => RoventNnuecHandoffTransferErrorV2::Busy,
                std::sync::TryLockError::Poisoned(_) => RoventNnuecHandoffTransferErrorV2::Poisoned,
            })?;
            original.parts_mut().0.clone()
        };
        let custody = RoventNnuecCollectionHandoffV2::try_transfer_session_custody(&handoff, &admission)?;
        self.transferred_handoff = Some(custody);
        self.handoff_transfer_error = None;
        Ok(true)
    }

    /// Collect only a thread that has already exited; never wait indefinitely
    /// or infer that a still-running worker has physically stopped.
    pub fn poll_worker(&mut self) -> bool {
        if self
            .worker_thread
            .as_ref()
            .is_some_and(JoinHandle::is_finished)
        {
            let handle = self.worker_thread.take().expect("finished worker retained");
            match handle.join() {
                Ok(exit) => self.worker_exit = Some(exit),
                Err(payload) => {
                    // The worker normally catches its own panic and returns
                    // its owner; an outer panic cannot prove physical exit.
                    std::mem::forget(payload);
                    self.worker_thread_panicked = true;
                    self.resource_error.get_or_insert_with(|| "teacher worker outer panic; original worker retained, physical completion unknown".into());
                }
            }
        }
        if self.worker_exit.is_some() && self.transferred_handoff.is_none() {
            if let Err(error) = self.try_transfer_joined_handoff() {
                self.handoff_transfer_error = Some(error);
            }
        }
        if (self.worker_exit.is_some() || self.worker_thread_panicked) && self.transferred_handoff.is_none()
            && self.recovered_handoff.is_none() && self.escrow_recovery_error.is_none()
        {
            if let Err(error) = self.try_recover_original_handoff() {
                self.escrow_recovery_error = Some(error);
            }
        }
        self.worker_exit.is_some() || self.worker_thread_panicked || self.worker_not_started
            || (self.worker_custody_poisoned && self.worker_thread.is_none())
    }

    pub fn observation(&self) -> serde_json::Value {
        let recovered_delivery = self.recovered_handoff.as_ref().map(|lease| {
            match lease.original().try_lock() {
                Ok(handoff) => match handoff.handoff_delivery() {
                    None => json!({"status":"pending"}),
                    Some(Ok(())) => json!({"status":"delivered"}),
                    Some(Err(error)) => json!({"status":"failed","error":format!("{error:?}")}),
                },
                Err(std::sync::TryLockError::WouldBlock) => json!({"status":"busy"}),
                Err(std::sync::TryLockError::Poisoned(_)) => json!({"status":"poisoned"}),
            }
        });
        let (pending_observation, pending_stopped) = match self.bridge.try_pending_stopped_observation() {
            Ok(Some(summary)) => ("available", Some(summary)),
            Ok(None) => ("busy", None),
            Err(_) => ("poisoned", None),
        };
        let (deferred_observation, deferred_roots) = match self.bridge.try_deferred_observations() {
            Ok(Some(roots)) => ("available", Some(roots)),
            Ok(None) => ("busy", None),
            Err(_) => ("poisoned", None),
        };
        let (resource_observation, resource) = match self.resource.try_lock() {
            Ok(state) => ("available", Some(state.observation())),
            Err(std::sync::TryLockError::WouldBlock) => ("busy", None),
            Err(std::sync::TryLockError::Poisoned(_)) => ("poisoned", None),
        };
        let (owner_observation, owner) = match self.owner.try_status() {
            Ok(Some(status)) => ("available", Some(status)),
            Ok(None) => ("busy", None),
            Err(_) => ("poisoned", None),
        };
        let worker_exit = match self.worker_exit.as_ref() {
            Some(RoventNnuecCollectionWorkerExitV2::NoAdmission) => "no_admission",
            Some(RoventNnuecCollectionWorkerExitV2::RetirementAndHandoffObserved) => {
                "retirement_and_handoff_observed"
            }
            Some(RoventNnuecCollectionWorkerExitV2::Deadline) => "deadline",
            Some(RoventNnuecCollectionWorkerExitV2::AdmissionCanceled) => "admission_canceled",
            Some(RoventNnuecCollectionWorkerExitV2::PhysicalAckDelivered) => {
                "physical_scope_ack_delivered"
            }
            Some(RoventNnuecCollectionWorkerExitV2::AggregateHandoffObserved) => {
                "physical_aggregate_handoff_observed"
            }
            Some(RoventNnuecCollectionWorkerExitV2::Panicked(_)) => "caught_panic",
            None if self.worker_not_started => "not_started",
            None if self.worker_custody_poisoned => "custody_poisoned",
            None if self.worker_thread_panicked => "outer_panic",
            None => "pending",
        };
        let worker_observation = self.worker.try_lock().ok().map(|worker| {
            let failures = worker.failure_observation();
            let (pump_unlabeled, pump_prepared, pump_committed, pump_failed) = worker.pump_ready_counts();
            let handoff = match worker.try_handoff_observation(|handoff| {
                let delivery = match handoff.handoff_delivery() {
                    None => json!({"status":"pending"}),
                    Some(Ok(())) => json!({"status":"delivered"}),
                    Some(Err(error)) => json!({"status":"failed","error":format!("{error:?}")}),
                };
                (handoff.collection_failures().iter().map(|error| format!("{error:?}")).collect::<Vec<_>>(), delivery)
            }) {
                Ok((failures,delivery)) => json!({"status":"available","incomplete":!failures.is_empty() || delivery["status"] != "delivered","failures":failures,"delivery":delivery}),
                Err(error) => json!({"status":format!("{error:?}"),"incomplete":"unknown","failures":null}),
            };
            json!({"primary_failure":failures.primary.map(|failure|failure.kind()),
                "additional_failures":failures.additional.iter().map(|failure|failure.kind()).collect::<Vec<_>>(),
                "omitted_failures":failures.omitted,
                "diagnostic_actors":worker.diagnostics().len(),"handoff_collection":handoff,
                "pump_unpublished_rows":worker.pending_pump_rows(),
                "pump_ready_unlabeled":pump_unlabeled,"pump_ready_prepared":pump_prepared,"pump_ready_committed":pump_committed,"pump_ready_failed":pump_failed})
        });
        json!({"schema":"rz-rovent-own-nnuec-lock-v2-run-observation/1",
            "native_error":self.native_failure.as_ref().map(|error|error.error.to_string()),
            "native_failure_transferred":self.native_failure_taken,
            "resource_error":self.resource_error,"resource_observation":resource_observation,"resource":resource,
            "teacher_worker_returned":self.worker_exit.is_some(),
            "teacher_worker_exit":worker_exit,
            "teacher_worker_running_or_unknown":!self.worker_not_started && (self.worker_thread.is_some() || self.worker_thread_panicked || self.worker_custody_poisoned
                || !matches!(self.worker_exit.as_ref(), Some(RoventNnuecCollectionWorkerExitV2::RetirementAndHandoffObserved))),
            "teacher_worker_thread_panicked":self.worker_thread_panicked,
            "teacher_worker_custody_poisoned":self.worker_custody_poisoned,
            "recovered_handoff_lease_retained":self.recovered_handoff.is_some(),
            "recovered_handoff_delivery":recovered_delivery,
            "escrow_recovery_error":self.escrow_recovery_error.as_ref().map(|error|error.to_string()),
            "escrow_recovery_state":if self.transferred_handoff.is_some() { "transferred" }
                else if self.recovered_handoff.is_some() { "recovered" }
                else if self.escrow_recovery_error.is_some() { "failed" }
                else if self.bridge.installed_admission_for_recovery().is_some() { "pending" }
                else { "no_installed_admission_observed" },
            "teacher_worker_observation":worker_observation,
            "collection_owner_observation":owner_observation,
            "collection_owner":owner.map(|status| json!({"admitted":status.admitted,"settled":status.settled,
                "pending":status.pending_count,"retired":status.pending_retired_count,
                "collection_incomplete_due_to_backpressure":status.collection_disabled_first_ticket.is_some() || !status.prior_disabled_first_tickets.is_empty(),
                "first_omitted_ticket":status.collection_disabled_first_ticket.as_ref().map(ticket_json),
                "disable_reason":status.collection_disabled_reason.map(|reason|format!("{reason:?}")),
                "prior_disabled_first_tickets":status.prior_disabled_first_tickets.iter().map(ticket_json).collect::<Vec<_>>(),
                "prior_disabled_reasons":status.prior_disabled_reasons.iter().map(|reason|format!("{reason:?}")).collect::<Vec<_>>(),
                "incomplete_capture":status.incomplete_capture_count,
                "failed_captures":status.retained_failed_captures,
                "failed_staged_inputs":status.retained_failed_staged,
                "pending_disabled_markers":status.pending_disabled_markers,
                "failure":status.failure.map(|error|format!("{error:?}")),
                "callback_lock_failure":status.callback_lock_failure})),
            "deferred_collection_roots_observation":deferred_observation,
            "pending_stopped_observation":pending_observation,
            "pending_stopped":pending_stopped.map(|summary| json!({
                "first":summary.first.map(|event| json!({
                    "admission_sha256":event.admission.raw_sha256(),
                    "cause":event.cause.as_ref().map(|cause| format!("{cause:?}")),
                    "reason":format!("{:?}",event.reason),
                })),
                "summarized":summary.summarized,
                "reasons":summary.reasons.into_iter().map(|(cause, reason, count)| json!({
                    "cause":cause.map(|cause| format!("{cause:?}")),
                    "reason":format!("{reason:?}"), "count":count,
                })).collect::<Vec<_>>(),
                "other":summary.other,
            })),
            "deferred_collection_roots":deferred_roots.map(|(roots, summarized, first_summarized, reasons, other_reasons)| json!({
                "retained":roots.into_iter().map(|(admission, reason)| json!({"admission_sha256":admission,"reason":format!("{reason:?}")})).collect::<Vec<_>>(),
                "summarized_count":summarized,
                "first_summarized_admission_sha256":first_summarized,
                "summarized_by_reason":reasons.into_iter().map(|(reason,count)| json!({"stage":format!("{:?}",reason.stage),"code":reason.code,"detail":reason.detail,"count":count})).collect::<Vec<_>>(),
                "other_summarized_reasons":other_reasons,
            })),
            "uci_handoff_custody":if self.transferred_handoff.is_some() { "leased_to_run" } else if self.recovered_handoff.is_some() { "recovered_from_escrow" } else { "unknown" },
            "uci_handoff_registry_released":false,
            "uci_handoff_transfer_error":self.handoff_transfer_error.map(|error|format!("{error:?}")),
            "physical_completion_claimed":false,"corpus_published":false,"training_executed":false})
    }

    /// Move the original native failure to a caller that will retain/recover
    /// it. Taking it never changes the original run into a success.
    pub fn take_native_failure(&mut self) -> Option<RoventOwnNnuecCollectionLaunchFailureV2> {
        let failure = self.native_failure.take();
        if failure.is_some() {
            self.native_failure_taken = true;
        }
        failure
    }

    /// Transfer actual teacher diagnostic bytes only after the worker thread
    /// has exited. A busy or poisoned owner stays retained as unknown.
    pub fn take_teacher_diagnostics(
        &mut self,
    ) -> Result<
        BTreeMap<String, rz_search::external_cpu::collection_teacher::TeacherDiagnostics>,
        ArenaError,
    > {
        check(
            self.worker_thread.is_none() && self.worker_exit.is_some()
                && !self.worker_thread_panicked && !self.worker_custody_poisoned,
            "Collection /2 teacher worker has not exited cleanly",
        )?;
        self.worker
            .try_lock()
            .map_err(|_| invalid("Collection /2 worker busy or poisoned"))
            .map(|mut worker| worker.take_diagnostics())
    }

    /// Read the actor retirement observations independently of diagnostic
    /// bytes. These values do not themselves authorize corpus publication.
    pub fn teacher_retirement_observations(
        &self,
    ) -> Result<
        BTreeMap<String, rz_uci::rovent_external_launch::RoventTeacherRetirementObservationV2>,
        ArenaError,
    > {
        self.worker
            .try_lock()
            .map_err(|_| invalid("Collection /2 worker busy or poisoned"))
            .map(|worker| worker.teacher_observations().clone())
    }

    /// The original handoff retains every Collection failure. Failure or a
    /// temporarily unavailable observation is never evidence of GPU failure.
    pub fn collection_failures(&self) -> Result<Vec<AdaptiveCollectionError>, ArenaError> {
        let worker = self
            .worker
            .try_lock()
            .map_err(|_| invalid("Collection /2 worker busy or poisoned"))?;
        worker
            .try_handoff_observation(|handoff| handoff.collection_failures().to_vec())
            .map_err(|error| invalid(format!("Collection /2 handoff observation: {error:?}")))
    }

    /// Give the caller the same collection owner only after its original
    /// worker observed the same fenced handoff and teacher retirement. This
    /// authorizes inspection/partial assembly, not final publication.
    pub fn closed_owner(&self) -> Result<RoventCollectionOwnerLockV2, ArenaError> {
        let worker = self.worker.try_lock().map_err(|_| invalid("Collection /2 worker busy or poisoned"))?;
        let delivered = worker.try_handoff_observation(|handoff| matches!(handoff.handoff_delivery(), Some(Ok(()))))
            .map_err(|error| invalid(format!("Collection /2 handoff delivery observation: {error:?}")))?;
        drop(worker);
        check(delivered, "Collection /2 handoff delivery pending or failed")?;
        check(
            self.collection_failures()?.is_empty(),
            "Collection /2 has incomplete handoff observations",
        )?;
        check(
            matches!(
                self.worker_exit.as_ref(),
                Some(RoventNnuecCollectionWorkerExitV2::RetirementAndHandoffObserved)
            ) && self
                .worker
                .try_lock()
                .is_ok_and(|worker| !worker.has_failure() && !worker.has_unpublished_pump_rows()),
            "Collection /2 teacher/worker retirement is not observed",
        )?;
        let status = self
            .owner
            .try_status()?
            .ok_or_else(|| invalid("Collection /2 owner busy"))?;
        check(
            status.admitted
                && status.settled
                && status.pending_count == 0
                && status.incomplete_capture_count == 0
                && status.retained_failed_captures == 0
                && status.retained_failed_staged == 0
                && status.pending_disabled_markers == Some(0)
                && status.failure.is_none()
                && !status.callback_lock_failure,
            "Collection /2 native observer is not settled",
        )?;
        Ok(self.owner.clone())
    }
}

impl Drop for RoventNnuecCollectionRunV2 {
    fn drop(&mut self) {
        // Only the explicit ownership transfer above releases an unstarted
        // outer owner. The resource evidence directory is never deleted.
        if self.pre_admission_failure_transferred {
            return;
        }
        let owner_resolved = self
            .owner
            .try_status()
            .ok()
            .flatten()
            .is_some_and(|status| {
                status.admitted
                    && status.settled
                    && status.pending_count == 0
                    && status.collection_disabled_first_ticket.is_none()
                    && status.prior_disabled_first_tickets.is_empty()
                    && status.pending_retired_count == 0
                    && status.incomplete_capture_count == 0
                    && status.retained_failed_captures == 0
                    && status.retained_failed_staged == 0
                    && status.pending_disabled_markers == Some(0)
                    && status.failure.is_none()
                    && !status.callback_lock_failure
            });
        // This only decides whether the Arena-owned worker can be dropped.
        // UCI's independent retained handoff slot remains unknown.
        let arena_worker_resolved = owner_resolved
            && self.bridge.try_has_deferred_observations().is_ok_and(|observation| observation == Some(false))
            && self.native_failure.is_none()
            && !self.native_failure_taken
            && self.resource_error.is_none()
            && !self.worker_custody_poisoned
            && self.commit_failure.is_none()
            && self.escrow_recovery_error.is_none()
            && self
                .collection_failures()
                .is_ok_and(|failures| failures.is_empty())
            && self
                .worker
                .try_lock()
                .is_ok_and(|worker| !worker.has_failure() && !worker.has_unpublished_pump_rows())
            && matches!(
                self.worker_exit.as_ref(),
                Some(RoventNnuecCollectionWorkerExitV2::RetirementAndHandoffObserved)
            );
        if !arena_worker_resolved {
            // A dropped observation must not release a still-running or
            // unverified owner. Keep the exact objects for process lifetime;
            // this is quarantine, not a physical-completion receipt.
            std::mem::forget(Arc::clone(&self.resource));
            std::mem::forget(Arc::clone(&self.original_lock));
            std::mem::forget(self.owner.clone());
            std::mem::forget(Arc::clone(&self.bridge));
            std::mem::forget(Arc::clone(&self.worker));
            if let Some(original) = self.commit_failure.take() {
                std::mem::forget(original);
            }
            if let Some(handle) = self.worker_thread.take() {
                std::mem::forget(handle);
            }
            if let Some(exit) = self.worker_exit.take() {
                std::mem::forget(exit);
            }
        }
    }
}

/// Load an explicitly pinned /2 manifest and start the actual Own NNUEC UCI path.
/// The plan's single-root limit is the bridge's current capability; it must
/// not be reported as multi-root Collection support.
pub fn run_rovent_own_nnuec_collection_lock_v2(
    arguments: Vec<String>,
    plan: RoventCollectionLockV2LaunchPlan,
    input_directory: &Dir,
    worker_deadline: Instant,
    cleanup_deadline: Instant,
) -> Result<RoventNnuecCollectionRunV2, ArenaError> {
    check(
        plan.maximum_roots == 1,
        "P/C Collection /2 bridge currently accepts one root",
    )?;
    check(
        Instant::now() < worker_deadline && worker_deadline <= cleanup_deadline,
        "P/C Collection /2 finite worker/cleanup deadline",
    )?;
    let bytes = read_pin(input_directory, &plan.manifest, 16 << 20)?;
    let manifest = Arc::new(RoventCorpusManifestLockV2::from_json(
        std::str::from_utf8(&bytes).map_err(|_| invalid("Collection /2 manifest UTF8"))?,
    )?);
    plan.validate_registration(&manifest)?;
    let (owner, observer) =
        RoventCollectionOwnerLockV2::new_shared(Arc::clone(&manifest), plan.metadata.clone())?;
    let roster = plan
        .teacher_slots
        .iter()
        .map(|entry| RoventTeacherRosterEntryV2 {
            actor_id: entry.actor_id.clone(),
            source_index: entry.source_index,
            collection_owner: entry.collection_owner,
        })
        .collect();
    let (bridge, worker) =
        RoventNnuecCollectionBridgeV2::new(owner.clone(), Arc::clone(&manifest), roster)?;
    // The consumer is not yet wired to offer, flush, and publish all original
    // rows. Keep the teacher row pump disabled until the bounded publisher is live.
    let worker = Arc::new(Mutex::new(worker));
    let resource = Arc::new(Mutex::new(resources::CollectionResourceCustody::prepare(
        input_directory,
        &plan.resource_plan,
    )?));
    let original_lock = Arc::new(Mutex::new(None));
    let admission_resource = Arc::clone(&resource);
    let admission_lock = Arc::clone(&original_lock);
    let stop = Arc::new(AtomicBool::new(false));
    let worker_stop = Arc::clone(&stop);
    let thread_worker = Arc::clone(&worker);
    // The output/cgroup resource gate can migrate only this startup thread.
    // Do not spawn any worker until that original gate has succeeded.
    let worker_slot = Arc::new(Mutex::new(None));
    let admission_worker_slot = Arc::clone(&worker_slot);
    let hooks: Arc<dyn rz_uci::rovent_bootstrap::nnuec_session::RoventNnuecCollectionHooksV2> =
        bridge.clone();
    let native = run_own_nnuec_collection_v2_with_resource_admission(
        arguments,
        plan.teacher_sources,
        Box::new(observer),
        hooks,
        move |lock| {
            let mut original = admission_lock.lock().map_err(|_| RoventLaunchError {
                stage: "resource_admission",
                detail: "Collection /2 lock custody poisoned".into(),
            })?;
            *original = Some(Arc::clone(lock));
            manifest.validate(lock).map_err(|error| RoventLaunchError {
                stage: "collection_manifest_v2",
                detail: error.to_string(),
            })?;
            {
                let mut resource = admission_resource.lock().map_err(|_| RoventLaunchError {
                    stage: "resource_admission",
                    detail: "Collection /2 resource custody poisoned".into(),
                })?;
                resource.admit_v2(lock).map_err(|error| RoventLaunchError {
                    stage: "resource_admission",
                    detail: error.to_string(),
                })?;
            }
            let mut slot = admission_worker_slot.lock().map_err(|_| RoventLaunchError {
                stage: "teacher_worker_start",
                detail: "Collection /2 worker handle custody poisoned".into(),
            })?;
            if slot.is_some() {
                return Err(RoventLaunchError { stage: "teacher_worker_start", detail: "Collection /2 worker already started".into() });
            }
            let handle = std::thread::Builder::new()
                .name("rovent-own-nnuec-collection-v2".into())
                .spawn(move || match thread_worker.lock() {
                    Ok(mut owner) => owner.run_until(worker_stop, worker_deadline),
                    Err(_) => panic!("teacher worker owner mutex poisoned before run"),
                })
                .map_err(|error| RoventLaunchError { stage: "teacher_worker_start", detail: error.to_string() })?;
            *slot = Some(handle);
            Ok(())
        },
    );
    stop.store(true, Ordering::Release);
    let (worker_thread, slot_poisoned) = match worker_slot.lock() {
        Ok(mut slot) => (slot.take(), false),
        Err(poison) => (poison.into_inner().take(), true),
    };
    let worker_not_started = worker_thread.is_none() && !slot_poisoned;
    let mut run = RoventNnuecCollectionRunV2 {
        resource,
        original_lock,
        owner,
        bridge,
        worker,
        worker_thread,
        worker_not_started,
        worker_exit: None,
        worker_thread_panicked: false,
        worker_custody_poisoned: slot_poisoned,
        transferred_handoff: None,
        recovered_handoff: None,
        escrow_recovery_error: None,
        handoff_transfer_error: None,
        commit_failure: None,
        native_failure: native.err(),
        native_failure_taken: false,
        pre_admission_failure_transferred: false,
        resource_error: None,
    };
    if let Ok(mut resource) = run.resource.lock() {
        if let Err(error) = resource.finish_live() {
            run.resource_error = Some(error.to_string());
        } else if run.native_failure.is_none() {
            let validated = run
                .original_lock
                .lock()
                .map_err(|_| invalid("Collection /2 original lock custody poisoned"))
                .and_then(|lock| {
                    lock.as_ref()
                        .ok_or_else(|| invalid("Collection /2 original lock missing"))
                        .and_then(|lock| resource.validate_success_v2(lock))
                });
            if let Err(error) = validated {
                run.resource_error = Some(error.to_string());
            }
        }
    } else {
        run.resource_error = Some("Collection /2 resource custody poisoned".into());
    }
    while !run.poll_worker() && Instant::now() < cleanup_deadline {
        std::thread::sleep(Duration::from_millis(2));
    }
    Ok(run)
}
