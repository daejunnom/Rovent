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
    collections::{BTreeMap, VecDeque},
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
    Captured { capsule: super::super::producer_v2::SubmittedTeacherQueryV2, failure: Box<FailedEvidence> },
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
    PendingMismatched {
        expected: TeacherJobId,
        completed: CompletedTeacherV2,
    },
    Completion(CompletedTeacherV2),
    CapturedMismatch(super::v2_session::SpeculativeSealedTeacherPolicyV2),
    CapturedJoinPanicked { sealed: super::v2_session::SpeculativeSealedTeacherPolicyV2, panic: Box<dyn std::any::Any + Send> },
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

impl FailedEvidence {
    /// Only explicit physical-unknown, poison, mismatch and panic evidence
    /// blocks route cleanup; a known semantic failure is still retained.
    pub(crate) fn has_uncertain_cleanup(&self) -> bool {
        match self {
            Self::Captured { failure, .. } => failure.has_uncertain_cleanup(),
            Self::Pending { .. } | Self::PendingMismatched { .. } | Self::Quarantined(_) | Self::CapturedMismatch(_)
            | Self::PreparationPanicked { .. } | Self::CapturedJoinPanicked { .. } => true,
            Self::SealRejected { cause, .. } => match cause {
                TeacherPolicySealFailureV2::Panicked(_) | TeacherPolicySealFailureV2::Validation(_) => true,
                TeacherPolicySealFailureV2::Native { kind, .. } => matches!(kind,
                    RoventTeacherPolicySealFailureKindV2::Poisoned | RoventTeacherPolicySealFailureKindV2::Invalid),
            },
            Self::BeforeCompletion | Self::SubmitPending(_) | Self::PendingCompleted { .. }
            | Self::Completion(_) | Self::Sealed(_) => false,
        }
    }
    /// Read-only description. The actual completion and authority remain
    /// with this failure; this text is never a substitute for either.
    pub(crate) fn diagnostic(&self) -> String {
        match self {
            Self::BeforeCompletion => "before_completion".into(),
            Self::Captured { failure, .. } => format!("captured_{}", failure.diagnostic()),
            Self::SubmitPending(_) => "submit_pending".into(),
            Self::Pending { .. } => "pending_original_job".into(),
            Self::PendingCompleted { .. } => "pending_completed".into(),
            Self::PendingMismatched { .. } => "pending_terminal_identity_mismatch".into(),
            Self::Completion(_) => "completed_unsealed".into(),
            Self::CapturedMismatch(_) => "captured_retirement_mismatch".into(),
            Self::CapturedJoinPanicked { .. } => "captured_retirement_join_panicked".into(),
            Self::SealRejected { cause, .. } => match cause {
                TeacherPolicySealFailureV2::Validation(error) => format!("seal_validation: {error}"),
                TeacherPolicySealFailureV2::Native { kind, error } => format!("seal_native {kind:?}: {error:?}"),
                TeacherPolicySealFailureV2::Panicked(_) => "seal_panicked".into(),
            },
            Self::Quarantined(_) => "seal_quarantined".into(),
            Self::Sealed(_) => "prepared_seal_rejected".into(),
            Self::PreparationPanicked { .. } => "preparation_panicked".into(),
        }
    }
}

struct Entry {
    row: RoventRetiredLockV2Record,
    deadline: Instant,
    stage: Stage,
}

/// A real captured input and, once consumed, its exact original retired row.
/// The capsule alone has no authority to produce a training target.
struct CapturedEntry {
    capsule: Option<super::super::producer_v2::SubmittedTeacherQueryV2>,
    ticket: rz_contracts::pals_adaptive::RoleTicketEnvelope,
    row: Option<RoventRetiredLockV2Record>,
    deadline: Instant,
    stage: CapturedStage,
}

enum CapturedStage {
    Queued,
    Unlabeled,
    TransitionUnknown,
    Pending { actor: String, id: TeacherJobId },
    Completed(CompletedTeacherV2),
    Sealed(super::v2_session::SpeculativeSealedTeacherPolicyV2),
    Prepared(PreparedTeacherPolicyLockV2),
    Failed { error: ArenaError, evidence: FailedEvidence },
}

/// Observations only; stage counts grant no completion or target authority.
#[derive(Default)]
pub(crate) struct PumpStageCountsV2 {
    pub queued: usize,
    pub unlabeled: usize,
    pub transition_unknown: usize,
    pub pending: usize,
    pub completed: usize,
    pub sealed: usize,
    pub prepared: usize,
    pub failed: usize,
}

/// Serial submission deliberately limits this worker to one outstanding
/// teacher job. Overflow returns the unmodified row to its owner.
pub(crate) struct RetiredRowPumpV2 {
    session_owner: TeacherPumpPermitV2,
    capacity: usize,
    entries: VecDeque<Entry>,
    captured: VecDeque<CapturedEntry>,
    seen_retired: BTreeMap<rz_contracts::RequestId, (rz_uci::pals_adaptive::RoventCollectionRootAdmissionV2,
        std::sync::Arc<()>, super::super::RoventSubmittedInputSeal, bool)>,
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
    captured: VecDeque<CapturedEntry>,
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
        for entry in &mut self.captured {
            match &mut entry.stage {
                CapturedStage::Pending { actor, id } => {
                    let expected = *id;
                    let Some(completed) = session.take_completed_actor(&self.permit, actor)? else { return Ok(false); };
                    entry.stage = CapturedStage::Failed { error: invalid("captured terminal retained during custody drain"),
                        evidence: if completed.dispatch.id() == expected {
                            FailedEvidence::PendingCompleted { id: expected, completed }
                        } else {
                            FailedEvidence::PendingMismatched { expected, completed }
                        } };
                    return Ok(true);
                }
                CapturedStage::Failed { evidence: FailedEvidence::Pending { actor, id }, .. } => {
                    let expected = *id;
                    let Some(completed) = session.take_completed_actor(&self.permit, actor)? else { return Ok(false); };
                    if let CapturedStage::Failed { evidence, .. } = &mut entry.stage {
                        *evidence = if completed.dispatch.id() == expected {
                            FailedEvidence::PendingCompleted { id: expected, completed }
                        } else {
                            FailedEvidence::PendingMismatched { expected, completed }
                        };
                    }
                    return Ok(true);
                }
                _ => {}
            }
        }
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
                        *evidence = if completed.dispatch.id() == id {
                            FailedEvidence::PendingCompleted { id, completed }
                        } else {
                            FailedEvidence::PendingMismatched { expected: id, completed }
                        };
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
            captured: VecDeque::new(),
            seen_retired: BTreeMap::new(),
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
        if self.seen_retired.len() >= 65_536 || self.seen_retired.contains_key(&row.ticket().id) {
            return Err(Box::new((invalid("retired teacher ticket repeated or ledger full"), row)));
        }
        if let Some(captured) = self.captured.iter_mut().find(|captured| captured.ticket.id == row.ticket().id) {
            if captured.row.is_some() || captured.ticket != *row.ticket()
                || captured.capsule.as_ref().is_some_and(|capsule| capsule.matches_original_row(&row).is_err()) {
                return Err(Box::new((invalid("captured teacher input and retired row differ or repeat"), row)));
            }
            self.seen_retired.insert(row.ticket().id,
                (row.admission.clone(), std::sync::Arc::clone(&row.producer_owner), row.submitted_input_seal(), true));
            captured.row = Some(row);
            return Ok(());
        }
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
        self.seen_retired.insert(row.ticket().id,
            (row.admission.clone(), std::sync::Arc::clone(&row.producer_owner), row.submitted_input_seal(), false));
        self.entries.push_back(Entry {
            row,
            deadline,
            stage: Stage::Queued,
        });
        Ok(())
    }

    /// A late capsule may be discarded as a duplicate only when the real
    /// retired path already retained the exact original producer identity.
    /// This never dispatches a teacher or establishes completion.
    pub(crate) fn is_exact_retired_duplicate(
        &self, capsule: &super::super::producer_v2::SubmittedTeacherQueryV2,
    ) -> bool {
        let ticket = capsule.ticket();
        self.seen_retired.get(&ticket.id).is_some_and(|(admission, producer, seal, via_captured)|
            !*via_captured && capsule.matches_original_identity(admission, producer, seal).is_ok())
    }

    pub(crate) fn can_enqueue_row(&self, row: &RoventRetiredLockV2Record) -> bool {
        self.entries.len() < self.capacity || self.captured.iter().any(|entry| entry.row.is_none() && entry.ticket.id == row.ticket().id)
    }

    pub(crate) fn overflow_count(&self) -> u64 {
        self.overflow_count
    }
    pub(crate) fn len(&self) -> usize {
        self.entries.len().saturating_add(self.captured.len())
    }

    /// Preserve one original captured query for a possible early teacher job.
    /// A capsule is never acceptance, retirement or a target.
    pub(crate) fn enqueue_captured(
        &mut self,
        capsule: super::super::producer_v2::SubmittedTeacherQueryV2,
    ) -> Result<(), Box<(ArenaError, super::super::producer_v2::SubmittedTeacherQueryV2)>> {
        let seal = capsule.input_seal();
        if let Some((admission, producer, original, via_captured)) = self.seen_retired.get(&seal.ticket.id) {
            // A captured-path job could be active: keep the duplicate as
            // unknown rather than discarding possible physical evidence.
            if *via_captured {
                return Err(Box::new((invalid("captured-path duplicate remains in custody"), capsule)));
            }
            // The legacy retired path owns this exact input. Avoid a duplicate
            // teacher job; no new target or completion is claimed.
            return match capsule.matches_original_identity(admission, producer, original) {
                Ok(()) => Ok(()),
                Err(error) => Err(Box::new((error, capsule))),
            };
        }
        if self.captured.len() >= self.capacity || !capsule.admission().same_original_admission(&self.admission)
            || seal.ticket.original_deadline.clock != self.clock_domain
            || self.captured.iter().any(|entry| entry.ticket.id == seal.ticket.id) {
            return Err(Box::new((invalid("captured teacher query capacity/root/duplicate"), capsule)));
        }
        let Some(deadline) = self.clock_origin.checked_add(Duration::from_nanos(seal.ticket.original_deadline.at.0)) else {
            return Err(Box::new((invalid("captured teacher query deadline overflow"), capsule)));
        };
        if self.captured.try_reserve(1).is_err() {
            return Err(Box::new((invalid("captured teacher query allocation failed"), capsule)));
        }
        self.captured.push_back(CapturedEntry { capsule: Some(capsule), ticket: seal.ticket, row: None, deadline: deadline.min(self.observation_deadline), stage: CapturedStage::Queued });
        Ok(())
    }

    /// Observe transitions of the same original front item without moving it.
    pub(crate) fn has_uncertain_failure(&self) -> bool {
        self.pending_failure || self.entries.iter().any(|entry| match &entry.stage {
            Stage::TransitionUnknown | Stage::Pending { .. } => true,
            Stage::Failed { evidence, .. } => evidence.has_uncertain_cleanup(),
            _ => false,
        }) || self.captured.iter().any(|entry| match &entry.stage {
            CapturedStage::TransitionUnknown | CapturedStage::Pending { .. } => true,
            CapturedStage::Failed { evidence, .. } => evidence.has_uncertain_cleanup(),
            _ => false,
        })
    }

    pub(crate) fn captured_len(&self) -> usize { self.captured.len() }
    pub(crate) fn retired_len(&self) -> usize { self.entries.len() }

    pub(crate) fn has_captured_completed(&self) -> bool {
        self.captured.iter().any(|entry| matches!(entry.stage, CapturedStage::Completed(_)))
    }

    pub(crate) fn front_pending(&self) -> bool {
        self.entries.front().is_some_and(|entry| matches!(entry.stage, Stage::Pending { .. }))
    }

    pub(crate) fn front_completed(&self) -> bool {
        self.entries.front().is_some_and(|entry| matches!(entry.stage, Stage::Completed(_)))
    }

    pub(crate) fn stage_counts(&self) -> PumpStageCountsV2 {
        let mut counts = PumpStageCountsV2::default();
        for entry in &self.entries {
            match &entry.stage {
                Stage::Queued => counts.queued += 1,
                Stage::Unlabeled => counts.unlabeled += 1,
                Stage::TransitionUnknown => counts.transition_unknown += 1,
                Stage::Pending { .. } => counts.pending += 1,
                Stage::Completed(_) => counts.completed += 1,
                Stage::Sealed(_) => counts.sealed += 1,
                Stage::Prepared(_) => counts.prepared += 1,
                Stage::Failed { .. } => counts.failed += 1,
            }
        }
        for entry in &self.captured {
            match &entry.stage {
                CapturedStage::Queued => counts.queued += 1,
                CapturedStage::Unlabeled => counts.unlabeled += 1,
                CapturedStage::TransitionUnknown => counts.transition_unknown += 1,
                CapturedStage::Pending { .. } => counts.pending += 1,
                CapturedStage::Completed(_) => counts.completed += 1,
                CapturedStage::Sealed(_) => counts.sealed += 1,
                CapturedStage::Prepared(_) => counts.prepared += 1,
                CapturedStage::Failed { .. } => counts.failed += 1,
            }
        }
        counts
    }

    /// One bounded state transition. Must run outside the producer mutex; seal
    /// must occur while the original root authority is still live. A failure
    /// remains in custody and is not converted to a policy label.
    pub(crate) fn poll(
        &mut self,
        session: &mut RoventCollectionTeacherSessionV2,
        allow_submission: bool,
    ) -> Result<(), ArenaError> {
        session.validate_pump_owner(&self.session_owner)?;
        if self.pending_failure {
            self.drain_pending_terminal(session)?;
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
                Some(_) if !allow_submission => Stage::Queued,
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

    /// Poll at most one captured job transition under the same original
    /// session permit. A captured seal never substitutes for native retirement.
    pub(crate) fn poll_captured(
        &mut self,
        session: &mut RoventCollectionTeacherSessionV2,
        allow_submission: bool,
    ) -> Result<(), ArenaError> {
        session.validate_pump_owner(&self.session_owner)?;
        // Terminal cleanup is independent of root admission. Never seal or
        // dispatch a new question from a failed pending entry.
        if let Some(entry) = self.captured.iter_mut().find(|entry| matches!(entry.stage,
            CapturedStage::Failed { evidence: FailedEvidence::Pending { .. }, .. })) {
            if let CapturedStage::Failed { evidence: FailedEvidence::Pending { actor, id }, .. } = &mut entry.stage {
                let expected = *id;
                if let Some(completed) = session.take_completed_actor(&self.session_owner, actor)? {
                    if let CapturedStage::Failed { evidence, .. } = &mut entry.stage {
                        *evidence = if completed.dispatch.id() == expected {
                            FailedEvidence::PendingCompleted { id: expected, completed }
                        } else {
                            FailedEvidence::PendingMismatched { expected, completed }
                        };
                    }
                }
            }
            return Ok(());
        }
        let allow_submission = allow_submission && !self.pending_failure;
        let index = if self.pending_failure {
            self.captured.iter().position(|entry| matches!(entry.stage, CapturedStage::Pending { .. }))
        } else {
            self.captured.iter().position(|entry| matches!(entry.stage, CapturedStage::Completed(_)))
                .or_else(|| self.captured.iter().position(|entry| entry.row.is_some() && matches!(entry.stage, CapturedStage::Sealed(_))))
                .or_else(|| self.captured.iter().position(|entry| matches!(entry.stage, CapturedStage::Pending { .. })))
                .or_else(|| if allow_submission { self.captured.iter().position(|entry| matches!(entry.stage, CapturedStage::Queued)) } else { None })
        };
        let Some(index) = index else { return Ok(()); };
        let entry = &mut self.captured[index];
        let stage = std::mem::replace(&mut entry.stage, CapturedStage::TransitionUnknown);
        entry.stage = match stage {
            CapturedStage::Queued => {
                let capsule = entry.capsule.as_ref().expect("queued capture retains capsule");
                let actor = capsule.input_seal().teacher_actor_id;
                match actor {
                    None => CapturedStage::Unlabeled,
                    Some(_) if !capsule.has_policy_candidates() => CapturedStage::Unlabeled,
                    Some(_) if capsule.is_canceled() => CapturedStage::Failed {
                        error: invalid("captured teacher request canceled before submission"),
                        evidence: FailedEvidence::BeforeCompletion,
                    },
                    Some(_) if !allow_submission => CapturedStage::Queued,
                    Some(_) if Instant::now() >= entry.deadline => CapturedStage::Failed {
                        error: invalid("captured teacher deadline expired before submission"), evidence: FailedEvidence::BeforeCompletion,
                    },
                    Some(actor) => match session.submit_captured_query(&self.session_owner, capsule, self.constraint, self.soft_quantum) {
                        Ok(TeacherSubmitOutcomeV2::Submitted(id)) => CapturedStage::Pending { actor, id },
                        Ok(TeacherSubmitOutcomeV2::Pending(_)) => CapturedStage::Queued,
                        Err(error) => CapturedStage::Failed { error, evidence: FailedEvidence::BeforeCompletion },
                    },
                }
            }
            CapturedStage::Pending { actor, id } => {
                let canceled = entry.capsule.as_ref().is_some_and(|capsule| capsule.is_canceled());
                if canceled {
                    match session.cancel_pending_ticket(&self.session_owner, &entry.ticket) {
                        Ok(super::v2_session::TeacherTicketCancelOutcomeV2::Requested
                            | super::v2_session::TeacherTicketCancelOutcomeV2::AlreadyRequested) => {},
                        Ok(super::v2_session::TeacherTicketCancelOutcomeV2::NotPending) => {
                            entry.stage = CapturedStage::Failed { error: invalid("captured canceled teacher original pending missing"),
                                evidence: FailedEvidence::Pending { actor, id } };
                            return Ok(());
                        },
                        Err(error) => {
                            entry.stage = CapturedStage::Failed { error, evidence: FailedEvidence::Pending { actor, id } };
                            return Ok(());
                        }
                    }
                }
                match session.take_completed_actor(&self.session_owner, &actor) {
                    Ok(Some(completed)) if completed.dispatch.id() != id => CapturedStage::Failed {
                        error: invalid("captured teacher terminal identity mismatch"),
                        evidence: FailedEvidence::PendingMismatched { expected: id, completed },
                    },
                    Ok(Some(completed)) if canceled || !allow_submission || Instant::now() >= entry.deadline => CapturedStage::Failed {
                        error: invalid("captured teacher terminal arrived after closure or deadline"),
                        evidence: FailedEvidence::PendingCompleted { id, completed },
                    },
                    Ok(Some(completed)) => CapturedStage::Completed(completed),
                    Ok(None) if Instant::now() >= entry.deadline => CapturedStage::Failed {
                        error: invalid("captured teacher terminal missing at deadline"),
                        evidence: FailedEvidence::Pending { actor, id },
                    },
                    Ok(None) => CapturedStage::Pending { actor, id },
                    Err(error) => CapturedStage::Failed { error, evidence: FailedEvidence::Pending { actor, id } },
                }
            }
            CapturedStage::Completed(completed) if entry.capsule.as_ref().is_some_and(|capsule| capsule.is_canceled())
                || !allow_submission || Instant::now() >= entry.deadline => CapturedStage::Failed {
                error: invalid("captured teacher completion not sealed before closure"),
                evidence: FailedEvidence::Completion(completed),
            },
            CapturedStage::Completed(completed) => {
                let capsule = entry.capsule.take().expect("completed capture retains capsule");
                match session.seal_captured_completion(&self.session_owner, capsule, completed) {
                    Ok(sealed) => CapturedStage::Sealed(sealed),
                    Err(rejected) => {
                        let (capsule, rejection) = *rejected;
                        entry.capsule = Some(capsule);
                        match rejection {
                            TeacherPolicySealRejectionV2::Rejected { cause, completed } => {
                                if matches!(&cause, TeacherPolicySealFailureV2::Native { kind: RoventTeacherPolicySealFailureKindV2::Busy, .. })
                                    && Instant::now() < entry.deadline {
                                    CapturedStage::Completed(completed)
                                } else {
                                    if matches!(&cause, TeacherPolicySealFailureV2::Panicked(_)) { self.pending_failure = true; }
                                    CapturedStage::Failed { error: invalid("captured teacher seal rejected"), evidence: FailedEvidence::SealRejected { cause, completed } }
                                }
                            }
                            TeacherPolicySealRejectionV2::Quarantined(custody) => {
                                self.pending_failure = true;
                                CapturedStage::Failed { error: invalid("captured teacher seal quarantined"), evidence: FailedEvidence::Quarantined(custody) }
                            },
                        }
                    }
                }
            }
            CapturedStage::Sealed(sealed) if entry.row.is_some() => {
                let row = entry.row.as_ref().expect("original retired row checked");
                match sealed.join_original_row(row) {
                    Ok(sealed) => match prepare_teacher_policy_lock_v2(row, sealed) {
                        Ok(prepared) => CapturedStage::Prepared(prepared),
                        Err(rejected) => {
                            let super::lock_v2_policy::PrepareTeacherPolicyFailureV2 { error, sealed, panic } = *rejected;
                            CapturedStage::Failed { error, evidence: match panic {
                                Some(panic) => { self.pending_failure = true; FailedEvidence::PreparationPanicked { sealed, panic } },
                                None => FailedEvidence::Sealed(sealed),
                            }}
                        }
                    },
                    Err(rejected) => {
                        let (cause, sealed) = *rejected;
                        match cause {
                            super::v2_session::SpeculativeTeacherJoinFailureV2::Validation(error) =>
                                CapturedStage::Failed { error, evidence: FailedEvidence::CapturedMismatch(sealed) },
                            super::v2_session::SpeculativeTeacherJoinFailureV2::Panicked(panic) => {
                                self.pending_failure = true;
                                CapturedStage::Failed { error: invalid("captured teacher join panicked"),
                                    evidence: FailedEvidence::CapturedJoinPanicked { sealed, panic } }
                            }
                        }
                    }
                }
            }
            unchanged => unchanged,
        };
        Ok(())
    }

    /// During failure/shutdown, retrieve only an already-submitted terminal.
    /// The same session and permit retain the original dispatch. No row is
    /// issued, sealed, labelled or made successful by this operation.
    pub(crate) fn drain_pending_terminal(
        &mut self,
        session: &mut RoventCollectionTeacherSessionV2,
    ) -> Result<bool, ArenaError> {
        session.validate_pump_owner(&self.session_owner)?;
        let Some(entry) = self.entries.front_mut() else { return Ok(false); };
        match &mut entry.stage {
            Stage::Pending { actor, id } => {
                let expected = *id;
                let Some(completed) = session.take_completed_actor(&self.session_owner, actor)? else { return Ok(false); };
                let matched = completed.dispatch.id() == expected;
                entry.stage = Stage::Failed {
                    error: invalid(if matched {
                        "V2 teacher terminal retained after pump failure"
                    } else {
                        "V2 pending teacher dispatch identity mismatch"
                    }),
                    evidence: if matched {
                        FailedEvidence::PendingCompleted { id: expected, completed }
                    } else {
                        FailedEvidence::PendingMismatched { expected, completed }
                    },
                };
                Ok(true)
            }
            Stage::Failed { evidence: FailedEvidence::Pending { actor, id }, .. } => {
                let expected = *id;
                let Some(completed) = session.take_completed_actor(&self.session_owner, actor)? else { return Ok(false); };
                if let Stage::Failed { evidence, .. } = &mut entry.stage {
                    *evidence = if completed.dispatch.id() == expected {
                        FailedEvidence::PendingCompleted { id: expected, completed }
                    } else {
                        FailedEvidence::PendingMismatched { expected, completed }
                    };
                }
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    /// Classify only an unissued front row after the worker observed revoke.
    /// It remains a failed/partial original row, never a teacher target.
    pub(crate) fn retire_unissued_after_revoke(&mut self) {
        for entry in &mut self.captured {
            if matches!(entry.stage, CapturedStage::Queued) {
                entry.stage = if entry.capsule.as_ref().is_some_and(|capsule|
                    capsule.input_seal().teacher_actor_id.is_none() || !capsule.has_policy_candidates()) {
                    CapturedStage::Unlabeled
                } else {
                    CapturedStage::Failed { error: invalid("captured teacher request unissued after root revoke"),
                        evidence: FailedEvidence::BeforeCompletion }
                };
            }
        }
        if let Some(entry) = self.entries.front_mut() {
            if matches!(entry.stage, Stage::Queued) {
                entry.stage = if entry.row.submitted_input_seal().teacher_actor_id.is_none()
                    || entry.row.record.0.input.legal_moves.is_empty() {
                    Stage::Unlabeled
                } else {
                    Stage::Failed {
                        error: invalid("V2 teacher request unissued after root revoke"),
                        evidence: FailedEvidence::BeforeCompletion,
                    }
                };
            }
        }
    }

    /// Prepared targets may be committed by the original producer only after
    /// the caller establishes callback quiescence. This function is custody
    /// transfer alone and does not establish quiescence or physical completion.
    pub(crate) fn take_ready(&mut self) -> Option<PumpItemV2> {
        // A captured query is useful only after the exact NN result retired.
        // Retain missing rows and unfinished evidence with this pump owner.
        if let Some(index) = self.captured.iter().position(|entry| entry.row.is_some()
            && matches!(entry.stage, CapturedStage::Unlabeled | CapturedStage::Prepared(_) | CapturedStage::Failed { .. })
            && !matches!(entry.stage, CapturedStage::Failed { evidence: FailedEvidence::Pending { .. }, .. })) {
            let entry = self.captured.remove(index).expect("captured item checked");
            let row = entry.row.expect("retired row checked");
            return Some(match entry.stage {
                CapturedStage::Unlabeled => PumpItemV2::Unlabeled { row },
                CapturedStage::Prepared(evidence) => PumpItemV2::Prepared { row, evidence: Some(evidence), sealed: None },
                CapturedStage::Failed { error, evidence } => PumpItemV2::Failed { row, error,
                    evidence: match entry.capsule {
                        Some(capsule) => FailedEvidence::Captured { capsule, failure: Box::new(evidence) },
                        None => evidence,
                    } },
                _ => unreachable!("captured readiness checked"),
            });
        }
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
            captured: self.captured,
        }
    }
}
