//! Original P-C root worker acquisition and its one retained attempt.
//!
//! A successful claim only authorizes this attempt. Neither claim settlement
//! nor a spawned thread establishes teacher, native, or GPU completion.
use super::*;

/// Install a fixed original slot before requesting any operation authority.
pub(super) fn ensure_pc_worker_operation_slot_v3(
    supervisor: &PcWorkerSupervisorV3,
    recovery: &mut PcRootRecoveryV3,
    stage_owner: &Arc<Mutex<TeacherSessionStagingV3>>,
) -> Result<Option<Arc<Mutex<PcOperationSlotV3>>>, &'static str> {
    const INDEX: usize = 1;
    if let Some(original) = &recovery.operation_slots[INDEX] {
        return Ok(Some(Arc::clone(original)));
    }
    let stage = match stage_owner.try_lock() {
        Ok(stage) => stage,
        Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
        Err(_) => return Err("P-C V3 stage poisoned"),
    };
    let Some(proof) = PcReceivedStageProofV3::try_from_guards(supervisor, recovery, stage_owner, &stage)? else {
        return Ok(None);
    };
    let identity = Arc::clone(&recovery.operation_identities[INDEX]);
    let admission = recovery.admission.clone();
    let owner = recovery.original_self.clone();
    let registration = supervisor.bridge.capture_registration().clone();
    let slot = Arc::new_cyclic(|original_self| Mutex::new(PcOperationSlotV3 {
        original_self: original_self.clone(), kind: PcOperationKindV3::NativeWorker,
        identity, admission, recovery: owner, registration,
        proof: Some(proof), claim: None, generation: None,
        route_destination: None, route_result: None, route_unknown: false,
        worker_destination: None, worker_failure_details: None,
        worker_undispatched: None, worker_unknown: false,
    }));
    recovery.operation_slots[INDEX] = Some(Arc::clone(&slot));
    Ok(Some(slot))
}

pub(super) enum PcWorkerEffectDriveV3 {
    Pending,
    Finished,
    Failed,
    Aborted,
    /// Caller must drop this original task outside the recovery guard and
    /// record its actual disposition before reobserving the same claim.
    DropUnstartedTask(RoventPcRootWorkerTaskV3),
}

/// Drive one original worker acquisition. The task factory is entered only
/// after the authority claim succeeds; reobservation passes `None`.
pub(super) fn drive_pc_worker_effect_v3<F>(
    supervisor: &PcWorkerSupervisorV3,
    recovery: &mut PcRootRecoveryV3,
    slot_owner: &Arc<Mutex<PcOperationSlotV3>>,
    task_factory: Option<F>,
) -> PcWorkerEffectDriveV3
where
    F: FnOnce() -> RoventPcRootWorkerTaskV3,
{
    use super::super::game_authority_v3::GameStepV3 as G;
    use PcWorkerEffectDriveV3 as D;

    let mut slot = match slot_owner.try_lock() {
        Ok(slot) => slot,
        Err(std::sync::TryLockError::WouldBlock) => return D::Pending,
        Err(_) => return D::Failed,
    };
    if slot.kind != PcOperationKindV3::NativeWorker
        || !Weak::ptr_eq(&slot.original_self, &Arc::downgrade(slot_owner))
        || !Weak::ptr_eq(&slot.recovery, &recovery.original_self)
        || !recovery.operation_slots[1].as_ref().is_some_and(|original| Arc::ptr_eq(original, slot_owner))
        || !Arc::ptr_eq(&slot.identity, &recovery.operation_identities[1])
        || !slot.admission.same_original_admission(&recovery.admission)
        || !slot.registration.same_original_owner(supervisor.bridge.capture_registration())
    {
        return D::Failed;
    }
    let mut proof = None;
    let mut claim = None;
    // Catch the whole period in which proof/claim are temporarily held here;
    // both are restored below even if the factory or producer panics.
    let attempted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        proof = slot.proof.take();
        claim = slot.claim.take();
        let Some(proof) = proof.as_ref() else { return D::Failed; };
        let mut newly_claimed = false;
        if claim.is_none() {
            if task_factory.is_none() {
                return D::Pending;
            }
            if slot.worker_destination.is_some()
                || slot.worker_failure_details.is_some()
                || slot.worker_undispatched.is_some()
                || slot.worker_unknown
                || recovery.worker.is_some()
                || recovery.worker_key.is_some()
                || recovery.worker_spawn_failure.is_some()
                || recovery.worker_failure_task_disposition.is_some()
            {
                return D::Failed;
            }
            match supervisor.bridge.game_authority().try_claim_pc_operation(
                proof, &slot, supervisor, recovery, &mut claim,
            ) {
                G::Ready => newly_claimed = true,
                G::Busy => return D::Pending,
                G::Aborted | G::Closed => return D::Aborted,
                _ => return D::Failed,
            }
        }
        let Some(original) = claim.as_mut() else { return D::Failed; };
        if newly_claimed {
            slot.generation = Some(original.generation());
            let Some(factory) = task_factory else {
                slot.worker_unknown = true;
                return D::Pending;
            };
            let task = factory();
            let admission = slot.admission.clone();
            match RoventPcCollectionWorkerV3::spawn_into(
                admission, task, &mut slot.worker_destination,
            ) {
                Ok(()) => {}
                Err((_admission, task)) => {
                    // A previously occupied destination is not an original
                    // spawn result. Preserve the claim as unknown and return
                    // the undispatched task for out-of-lock disposal.
                    slot.worker_unknown = true;
                    recovery.worker_failure_task_disposition = Some(PcUnstartedTaskDispositionV3::Pending);
                    recovery.failure.get_or_insert_with(|| "P-C V3 worker destination was occupied".into());
                    return D::DropUnstartedTask(task);
                }
            }
            match slot.worker_destination.take() {
                Some(Err(failure)) => {
                    let same_admission = failure.admission.same_original_admission(&slot.admission);
                    let (details, task) = failure.into_details_and_task();
                    slot.worker_failure_details = Some(details);
                    recovery.worker_failure_task_disposition = Some(PcUnstartedTaskDispositionV3::Pending);
                    if !same_admission { slot.worker_unknown = true; }
                    recovery.failure.get_or_insert_with(|| if same_admission {
                        "P-C V3 worker spawn failed".into()
                    } else {
                        "P-C V3 worker spawn failure admission differs".into()
                    });
                    return D::DropUnstartedTask(task);
                }
                Some(Ok(worker)) => slot.worker_destination = Some(Ok(worker)),
                None => {
                    slot.worker_unknown = true;
                    return D::Pending;
                }
            }
        }
        // Do not settle the original attempt if another owner/result has
        // appeared in its recovery: retain both original slot and claim.
        if (slot.worker_failure_details.is_some() && recovery.worker_spawn_failure.is_some())
            || (matches!(slot.worker_destination.as_ref(), Some(Ok(_)))
                && (recovery.worker.is_some() || recovery.worker_key.is_some()))
        {
            slot.worker_unknown = true;
            return D::Pending;
        }
        match original.try_complete_pc_operation(proof, &slot, supervisor, recovery) {
            G::Ready => {}
            G::Busy | G::Unknown => return D::Pending,
            _ => return D::Failed,
        }
        // Only the original, already-settled claim can be released here.
        claim = None;
        if let Some(details) = slot.worker_failure_details.take() {
            recovery.worker_spawn_failure = Some(details);
            return D::Failed;
        }
        match slot.worker_destination.take() {
            Some(Ok(worker)) => {
                recovery.worker_key = Some(worker.recovery_key());
                recovery.worker = Some(worker);
                D::Finished
            }
            other => {
                slot.worker_destination = other;
                slot.worker_unknown = true;
                D::Pending
            }
        }
    }));

    slot.proof = proof;
    slot.claim = claim;
    match attempted {
        Ok(outcome) => outcome,
        Err(payload) => {
            slot.worker_unknown = true;
            retain_pc_panic_v3(&mut recovery.panic, payload);
            recovery.failure.get_or_insert_with(|| "P-C V3 worker acquisition outcome unknown".into());
            // If a failure carrying the original self-referential task made
            // it into the producer slot, separate it before returning so the
            // task can be disposed of without the recovery mutex held.
            match slot.worker_destination.take() {
                Some(Err(failure)) => {
                    let (details, task) = failure.into_details_and_task();
                    slot.worker_failure_details = Some(details);
                    recovery.worker_failure_task_disposition = Some(PcUnstartedTaskDispositionV3::Pending);
                    D::DropUnstartedTask(task)
                }
                other => {
                    slot.worker_destination = other;
                    D::Pending
                }
            }
        }
    }
}
