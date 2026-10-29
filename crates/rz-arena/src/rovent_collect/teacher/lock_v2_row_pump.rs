//! Bounded custody for original retired rows and actual teacher evidence.
//! This worker does not publish rows or infer physical retirement.
use super::lock_v2_policy::{prepare_teacher_policy_lock_v2, PreparedTeacherPolicyLockV2};
use super::v2_session::{
    CompletedTeacherV2, RoventCollectionTeacherSessionV2, SealedTeacherPolicyV2,
    TeacherPolicyQuarantineV2, TeacherPolicySealFailureV2, TeacherPolicySealRejectionV2,
    TeacherPumpPermitV2, TeacherSubmitOutcomeV2, TeacherSubmitPendingV2,
};
use super::*;
use rz_search::external_cpu::collection_teacher::TeacherJobId;
use rz_search::external_cpu::{ExternalGoConstraint, EXTERNAL_MIN_QUANTUM};
use rz_uci::rovent_external_launch::RoventTeacherPolicySealFailureKindV2;
use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

enum Stage {
    Queued,
    Unlabeled,
    TransitionUnknown,
    Pending {
        actor: String,
        id: TeacherJobId,
    },
    Completed(CompletedTeacherV2),
    Sealed(SealedTeacherPolicyV2),
    Prepared(PreparedTeacherPolicyLockV2),
    Failed {
        error: ArenaError,
        evidence: FailedEvidence,
    },
}

/// Failures carry the real source object, never a reconstructed receipt.
pub(crate) enum FailedEvidence {
    BeforeCompletion,
    SubmitPending(TeacherSubmitPendingV2),
    // The session may still own the pending worker job.
    Pending {
        actor: String,
        id: TeacherJobId,
    },
    PendingCompleted {
        id: TeacherJobId,
        completed: CompletedTeacherV2,
    },
    Completion(CompletedTeacherV2),
    SealRejected {
        cause: TeacherPolicySealFailureV2,
        completed: CompletedTeacherV2,
    },
    Quarantined(TeacherPolicyQuarantineV2),
    Sealed(SealedTeacherPolicyV2),
    PreparationPanicked {
        sealed: SealedTeacherPolicyV2,
        panic: Box<dyn std::any::Any + Send>,
    },
}

struct Entry {
    row: RoventRetiredLockV2Record,
    deadline: Instant,
    stage: Stage,
}

/// Serial submission deliberately limits this worker to one outstanding
/// teacher job. Overflow returns the unmodified row to its owner.
pub(crate) struct RetiredRowPumpV2 {
    session_owner: TeacherPumpPermitV2,
    capacity: usize,
    entries: VecDeque<Entry>,
    constraint: ExternalGoConstraint,
    soft_quantum: Duration,
    overflow_count: u64,
    pending_failure: bool,
    admission: rz_uci::pals_adaptive::RoventCollectionRootAdmissionV2,
    clock_origin: Instant,
    clock_domain: rz_contracts::ClockDomain,
    observation_deadline: Instant,
}

pub(crate) enum PumpItemV2 {
    Unlabeled {
        row: RoventRetiredLockV2Record,
    },
    Prepared {
        row: RoventRetiredLockV2Record,
        evidence: Option<PreparedTeacherPolicyLockV2>,
        sealed: Option<SealedTeacherPolicyV2>,
    },
    Committed {
        row: RoventRetiredLockV2Record,
        sealed: SealedTeacherPolicyV2,
    },
    Failed {
        row: RoventRetiredLockV2Record,
        error: ArenaError,
        evidence: FailedEvidence,
    },
}

/// Remaining custody can be handed to the session owner on shutdown; Pending
/// means the actual dispatch is still owned by that same session.
pub(crate) enum HeldPumpStageV2 {
    Queued,
    Unlabeled,
    TransitionUnknown,
    Pending {
        actor: String,
        id: TeacherJobId,
    },
    Completed(CompletedTeacherV2),
    Sealed(SealedTeacherPolicyV2),
    Prepared(PreparedTeacherPolicyLockV2),
    Failed {
        error: ArenaError,
        evidence: FailedEvidence,
    },
}

pub(crate) struct HeldPumpEntryV2 {
    pub(crate) row: RoventRetiredLockV2Record,
    pub(crate) stage: HeldPumpStageV2,
}

/// The one-shot permit must travel with outstanding rows; dropping a pump
/// never proves teacher retirement or licenses a replacement pump.
pub(crate) struct PumpCustodyV2 {
    pub(crate) permit: TeacherPumpPermitV2,
    pub(crate) entries: Vec<HeldPumpEntryV2>,
}

impl PumpCustodyV2 {
    /// Collect a late terminal from the same session without sealing or
    /// labeling it. The caller still must shut down and physically drain the
    /// original worker. A missing terminal is not a completed operation.
    pub(crate) fn drain_one(
        &mut self,
        session: &mut RoventCollectionTeacherSessionV2,
    ) -> Result<bool, ArenaError> {
        session.validate_pump_owner(&self.permit)?;
        for entry in &mut self.entries {
            match &mut entry.stage {
                HeldPumpStageV2::Pending { actor, .. } => {
                    let Some(completed) = session.take_completed_actor(&self.permit, actor)? else {
                        return Ok(false);
                    };
                    entry.stage = HeldPumpStageV2::Completed(completed);
                    return Ok(true);
                }
                HeldPumpStageV2::Failed {
                    evidence: FailedEvidence::Pending { actor, id },
                    ..
                } => {
                    let Some(completed) = session.take_completed_actor(&self.permit, actor)? else {
                        return Ok(false);
                    };
                    let id = *id;
                    if let HeldPumpStageV2::Failed { evidence, .. } = &mut entry.stage {
                        *evidence = FailedEvidence::PendingCompleted { id, completed };
                    }
                    return Ok(true);
                }
                _ => {}
            }
        }
        Ok(false)
    }
}

impl RetiredRowPumpV2 {
    pub(crate) fn new(
        session: &mut RoventCollectionTeacherSessionV2,
        capacity: usize,
        constraint: ExternalGoConstraint,
        soft_quantum: Duration,
    ) -> Result<Self, ArenaError> {
        check(
            capacity > 0 && capacity <= 64 && soft_quantum >= EXTERNAL_MIN_QUANTUM,
            "V2 row pump requires bounded capacity and the worker minimum quantum",
        )?;
        let root = session.original_admission().root_admission();
        let hard = root.hard_deadline();
        let clock = root.runtime_clock();
        let clock_domain = session.original_admission().clock_domain();
        check(
            hard.clock == clock_domain,
            "V2 row pump clock domain differs from root",
        )?;
        let clock_origin = clock.origin();
        let admission = session.original_admission().clone();
        let observation_deadline = clock_origin
            .checked_add(Duration::from_nanos(hard.at.0))
            .ok_or_else(|| invalid("V2 row pump root deadline cannot be represented"))?;
        // Claim after all fallible configuration validation; no replacement
        // pump can steal completion from this original session.
        let session_owner = session.claim_pump_owner()?;
        Ok(Self {
            session_owner,
            capacity,
            entries: VecDeque::new(),
            constraint,
            soft_quantum,
            overflow_count: 0,
            pending_failure: false,
            admission,
            clock_origin,
            clock_domain,
            observation_deadline,
        })
    }

    pub(crate) fn enqueue(
        &mut self,
        row: RoventRetiredLockV2Record,
    ) -> Result<(), Box<(ArenaError, RoventRetiredLockV2Record)>> {
        if self.entries.len() >= self.capacity {
            self.overflow_count = self.overflow_count.saturating_add(1);
            return Err(Box::new((invalid("V2 row pump is full"), row)));
        }
        let original_deadline = row.ticket().original_deadline;
        if !row.admission.same_original_admission(&self.admission)
            || original_deadline.clock != self.clock_domain
        {
            return Err(Box::new((
                invalid("V2 row pump row has a different original root or clock"),
                row,
            )));
        }
        let Some(deadline) = self
            .clock_origin
            .checked_add(Duration::from_nanos(original_deadline.at.0))
        else {
            return Err(Box::new((
                invalid("V2 row pump ticket deadline cannot be represented"),
                row,
            )));
        };
        let deadline = deadline.min(self.observation_deadline);
        self.entries.push_back(Entry {
            row,
            deadline,
            stage: Stage::Queued,
        });
        Ok(())
    }

    pub(crate) fn overflow_count(&self) -> u64 {
        self.overflow_count
    }
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    /// One bounded state transition. Must run outside the producer mutex; seal
    /// must occur while the original root authority is still live. A failure
    /// remains in custody and is not converted to a policy label.
    pub(crate) fn poll(
        &mut self,
        session: &mut RoventCollectionTeacherSessionV2,
    ) -> Result<(), ArenaError> {
        session.validate_pump_owner(&self.session_owner)?;
        if self.pending_failure {
            return Ok(());
        }
        let Some(entry) = self.entries.front_mut() else {
            return Ok(());
        };
        // A panic leaves an explicit unknown stage, never an unissued row.
        let current = std::mem::replace(&mut entry.stage, Stage::TransitionUnknown);
        entry.stage = match current {
            Stage::Queued => match entry.row.submitted_input_seal().teacher_actor_id {
                None => Stage::Unlabeled,
                Some(_) if entry.row.record.0.input.legal_moves.is_empty() => Stage::Unlabeled,
                Some(_) if Instant::now() >= entry.deadline => Stage::Failed {
                    error: invalid("V2 teacher observation deadline expired before submission"),
                    evidence: FailedEvidence::BeforeCompletion,
                },
                Some(actor) => {
                    match session.submit_retired(
                        &self.session_owner,
                        &entry.row,
                        self.constraint,
                        self.soft_quantum,
                    ) {
                        Ok(TeacherSubmitOutcomeV2::Submitted(id)) => Stage::Pending { actor, id },
                        Ok(TeacherSubmitOutcomeV2::Pending(pending)) => {
                            if Instant::now() < entry.deadline {
                                Stage::Queued
                            } else {
                                Stage::Failed {
                                    error: invalid(
                                        "V2 teacher startup observation deadline expired",
                                    ),
                                    evidence: FailedEvidence::SubmitPending(pending),
                                }
                            }
                        }
                        Err(error) => Stage::Failed {
                            error,
                            evidence: FailedEvidence::BeforeCompletion,
                        },
                    }
                }
            },
            Stage::Pending { actor, id } if Instant::now() >= entry.deadline => {
                self.pending_failure = true;
                Stage::Failed {
                    error: invalid(
                        "V2 pending teacher observation deadline expired; session drain required",
                    ),
                    evidence: FailedEvidence::Pending { actor, id },
                }
            }
            Stage::Pending { actor, id } => {
                match session.take_completed_actor(&self.session_owner, &actor) {
                    Ok(Some(completed)) => Stage::Completed(completed),
                    Ok(None) => Stage::Pending { actor, id },
                    Err(error) => {
                        self.pending_failure = true;
                        Stage::Failed {
                            error,
                            evidence: FailedEvidence::Pending { actor, id },
                        }
                    }
                }
            }
            Stage::Completed(completed) if Instant::now() >= entry.deadline => Stage::Failed {
                error: invalid("V2 teacher policy observation deadline expired before sealing"),
                evidence: FailedEvidence::Completion(completed),
            },
            Stage::Completed(completed) => {
                match session.seal_completed_policy(&self.session_owner, &entry.row, completed) {
                    Ok(sealed) => Stage::Sealed(sealed),
                    Err(rejected) => match *rejected {
                        TeacherPolicySealRejectionV2::Rejected { cause, completed } => {
                            if matches!(
                                &cause,
                                TeacherPolicySealFailureV2::Native {
                                    kind: RoventTeacherPolicySealFailureKindV2::Busy,
                                    ..
                                }
                            ) && Instant::now() < entry.deadline
                            {
                                Stage::Completed(completed)
                            } else {
                                if matches!(&cause, TeacherPolicySealFailureV2::Panicked(_)) {
                                    self.pending_failure = true;
                                }
                                Stage::Failed {
                                    error: invalid("V2 teacher policy sealing rejected"),
                                    evidence: FailedEvidence::SealRejected { cause, completed },
                                }
                            }
                        }
                        TeacherPolicySealRejectionV2::Quarantined(custody) => {
                            self.pending_failure = true;
                            Stage::Failed {
                                error: invalid("V2 teacher policy evidence quarantined"),
                                evidence: FailedEvidence::Quarantined(custody),
                            }
                        }
                    },
                }
            }
            Stage::Sealed(sealed) => match prepare_teacher_policy_lock_v2(&entry.row, sealed) {
                Ok(prepared) => Stage::Prepared(prepared),
                Err(rejected) => {
                    let super::lock_v2_policy::PrepareTeacherPolicyFailureV2 {
                        error,
                        sealed,
                        panic,
                    } = *rejected;
                    let evidence = if let Some(panic) = panic {
                        self.pending_failure = true;
                        FailedEvidence::PreparationPanicked { sealed, panic }
                    } else {
                        FailedEvidence::Sealed(sealed)
                    };
                    Stage::Failed { error, evidence }
                }
            },
            unchanged => unchanged,
        };
        Ok(())
    }

    /// Prepared targets may be committed by the original producer only after
    /// the caller establishes callback quiescence. This function is custody
    /// transfer alone and does not establish quiescence or physical completion.
    pub(crate) fn take_ready(&mut self) -> Option<PumpItemV2> {
        // A pending timeout still has a job in the original session. Keep its
        // row with the permit so late terminal drain can recover both.
        if matches!(
            &self.entries.front()?.stage,
            Stage::Failed {
                evidence: FailedEvidence::Pending { .. },
                ..
            }
        ) {
            return None;
        }
        if !matches!(
            self.entries.front()?.stage,
            Stage::Unlabeled | Stage::Prepared(_) | Stage::Failed { .. }
        ) {
            return None;
        }
        let entry = self.entries.pop_front().expect("front was present");
        match entry.stage {
            Stage::Unlabeled => Some(PumpItemV2::Unlabeled { row: entry.row }),
            Stage::Prepared(evidence) => Some(PumpItemV2::Prepared {
                row: entry.row,
                evidence: Some(evidence),
                sealed: None,
            }),
            Stage::Failed { error, evidence } => Some(PumpItemV2::Failed {
                row: entry.row,
                error,
                evidence,
            }),
            _ => unreachable!("front readiness checked"),
        }
    }

    pub(crate) fn into_custody(self) -> PumpCustodyV2 {
        let entries = self
            .entries
            .into_iter()
            .map(|entry| HeldPumpEntryV2 {
                row: entry.row,
                stage: match entry.stage {
                    Stage::Queued => HeldPumpStageV2::Queued,
                    Stage::Unlabeled => HeldPumpStageV2::Unlabeled,
                    Stage::TransitionUnknown => HeldPumpStageV2::TransitionUnknown,
                    Stage::Pending { actor, id } => HeldPumpStageV2::Pending { actor, id },
                    Stage::Completed(completed) => HeldPumpStageV2::Completed(completed),
                    Stage::Sealed(sealed) => HeldPumpStageV2::Sealed(sealed),
                    Stage::Prepared(prepared) => HeldPumpStageV2::Prepared(prepared),
                    Stage::Failed { error, evidence } => {
                        HeldPumpStageV2::Failed { error, evidence }
                    }
                },
            })
            .collect();
        PumpCustodyV2 {
            permit: self.session_owner,
            entries,
        }
    }
}
