//! Typed profiled-external R assembly and raw reported-work observations.
//!
//! Neither the current-root source nor a returned UCI terminal calibrates
//! foreign reported nodes/depth/CP/mate to native credit, selective depth,
//! Rules proof or a physical root-reuse fence. The real external owner keeps
//! dispatch, stop/drain and root/process fences independently.
use super::*;
use crate::external_cpu::{
    ExternalEnginePurpose, ProfiledExternalCompletion, ProfiledExternalDispatch,
    ProfiledExternalRootSource,
};
use rz_contracts::rovent::SearchRecipient;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FrontierExternalWork {
    pub dispatched_questions: usize,
    pub returned_terminal_packets: usize,
    pub failed_terminal_packets: usize,
    pub outstanding_questions: usize,
    /// Sum only of observed engine-reported samples, with no calibration.
    pub reported_nodes_known_sum: u64,
    pub reported_nodes_known_samples: usize,
    pub reported_nodes_unknown_samples: usize,
    /// Authenticated report metadata, not this controller's drain fence.
    pub reported_ready_samples: usize,
}

/// Original external-dispatch custody; it reserves no native or common node
/// credit and cannot be converted into a NativeReservation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrontierExternalReservation {
    owner: u64,
    slot: usize,
    execution: rz_contracts::ExecutionId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrontierExternalAcceptance {
    RecordedTerminal,
    AlreadyRecordedExact,
}

struct ExternalObservation {
    dispatch: ProfiledExternalDispatch,
    completion: Option<Weak<ProfiledExternalCompletion>>,
}
#[derive(Default)]
pub(super) struct ExternalFrontierJournal {
    rows: Vec<ExternalObservation>,
    work: FrontierExternalWork,
}
impl ExternalFrontierJournal {
    pub(super) fn snapshot(&self) -> FrontierExternalWork {
        self.work
    }
}

pub(super) fn validate_external_registration(
    registration: &FrontierRegistration,
    source: &ProfiledExternalRootSource,
    clock: ClockDomain,
) -> Result<(), FrontierError> {
    source
        .validate_current()
        .map_err(|_| FrontierError::MissingAcceptance)?;
    if !registration.mode.has_native_r()
        || registration.native_ledger.is_some()
        || source.purpose() != ExternalEnginePurpose::ResearchChecker
        || source.authority() != registration.authority
        || source.clock().domain != clock
        || registration.original_deadline.clock != clock
    {
        return Err(FrontierError::ForeignRoot);
    }
    Ok(())
}

impl RoventFrontier {
    pub fn external_source(&self) -> Option<&ProfiledExternalRootSource> {
        self.external_source.as_ref()
    }

    pub fn external_work(&self) -> Option<FrontierExternalWork> {
        self.external_source
            .as_ref()
            .map(|_| self.external.snapshot())
    }

    /// Actual owner mailbox admission is a different typed resource from the
    /// own-R ledger. There is no reservation/subtraction on common nodes here.
    pub fn record_external_dispatch(
        &mut self,
        source: &ProfiledExternalRootSource,
        dispatch: &ProfiledExternalDispatch,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<FrontierExternalReservation, FrontierError> {
        self.preflight_external_dispatch(source, clock, now)?;
        self.record_observed_external_dispatch(source, dispatch)
    }

    /// Check the original live gate before the concrete owner submits. This
    /// reserves no resource and issues no dispatch/completion capability. The
    /// same single writer must still journal its actual opaque dispatch.
    pub fn preflight_external_dispatch(
        &mut self,
        source: &ProfiledExternalRootSource,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<(), FrontierError> {
        self.check_control(clock, now)?;
        source
            .validate_current()
            .map_err(|_| FrontierError::MissingAcceptance)?;
        self.check_external_source(source)?;
        self.check_external_journal_capacity()?;
        // Allocation and overflow must be known before the foreign worker owns
        // a request. Retain the allocation for the single-writer journal.
        self.external
            .rows
            .try_reserve(1)
            .map_err(|_| FrontierError::Capacity("external reported-work journal"))?;
        self.external
            .work
            .dispatched_questions
            .checked_add(1)
            .ok_or(FrontierError::Invalid("external dispatch counter overflow"))?;
        self.external
            .work
            .outstanding_questions
            .checked_add(1)
            .ok_or(FrontierError::Invalid("external pending counter overflow"))?;
        Ok(())
    }

    /// Preserve a dispatch already admitted by the real private worker, even
    /// if its original deadline/revocation raced the return to this writer.
    /// The source/question/budget/deadline identity checks remain mandatory;
    /// this projection grants no new admission, credit or extended lifetime.
    pub fn record_observed_external_dispatch(
        &mut self,
        source: &ProfiledExternalRootSource,
        dispatch: &ProfiledExternalDispatch,
    ) -> Result<FrontierExternalReservation, FrontierError> {
        self.check_external_source(source)?;
        self.record_external_dispatch_projection(source, dispatch)
    }

    fn check_external_journal_capacity(&self) -> Result<(), FrontierError> {
        if self.external.rows.len() >= self.limits.maximum_native_receipts {
            Err(FrontierError::Capacity("external reported-work journal"))
        } else {
            Ok(())
        }
    }

    fn record_external_dispatch_projection(
        &mut self,
        source: &ProfiledExternalRootSource,
        dispatch: &ProfiledExternalDispatch,
    ) -> Result<FrontierExternalReservation, FrontierError> {
        self.check_external_dispatch(source, dispatch)?;
        if self
            .external
            .rows
            .iter()
            .any(|row| row.dispatch.scope().execution == dispatch.scope().execution)
        {
            return Err(FrontierError::Duplicate);
        }
        self.check_external_journal_capacity()?;
        self.external
            .rows
            .try_reserve(1)
            .map_err(|_| FrontierError::Capacity("external reported-work journal"))?;
        let slot = self.external.rows.len();
        let dispatched = self
            .external
            .work
            .dispatched_questions
            .checked_add(1)
            .ok_or(FrontierError::Invalid("external dispatch counter overflow"))?;
        let outstanding = self
            .external
            .work
            .outstanding_questions
            .checked_add(1)
            .ok_or(FrontierError::Invalid("external pending counter overflow"))?;
        self.external.rows.push(ExternalObservation {
            dispatch: dispatch.clone(),
            completion: None,
        });
        self.external.work.dispatched_questions = dispatched;
        self.external.work.outstanding_questions = outstanding;
        Ok(FrontierExternalReservation {
            owner: self.owner,
            slot,
            execution: dispatch.scope().execution,
        })
    }

    /// Observe the original immutable private-worker packet, including late
    /// failed/canceled results. This does not accept it as chess/teacher value.
    /// Source comparison remains possible after revocation; new admission does
    /// not. Exact replay requires the same original packet allocation.
    pub fn record_external_completion(
        &mut self,
        source: &ProfiledExternalRootSource,
        reservation: FrontierExternalReservation,
        completion: &Arc<ProfiledExternalCompletion>,
    ) -> Result<FrontierExternalAcceptance, FrontierError> {
        self.check_external_source(source)?;
        if reservation.owner != self.owner {
            return Err(FrontierError::ForeignHandle);
        }
        let row = self
            .external
            .rows
            .get(reservation.slot)
            .ok_or(FrontierError::ForeignHandle)?;
        if row.dispatch.scope().execution != reservation.execution
            || !completion.matches_submission(&row.dispatch)
            || completion.process_identity() != Some(source.process_identity())
        {
            return Err(FrontierError::MissingAcceptance);
        }
        self.check_external_dispatch(source, &row.dispatch)?;
        if let Some(original) = &row.completion {
            return if original.ptr_eq(&Arc::downgrade(completion)) {
                Ok(FrontierExternalAcceptance::AlreadyRecordedExact)
            } else {
                Err(FrontierError::Duplicate)
            };
        }
        let mut next = self.external.work;
        next.returned_terminal_packets =
            next.returned_terminal_packets
                .checked_add(1)
                .ok_or(FrontierError::Invalid(
                    "external completion counter overflow",
                ))?;
        next.outstanding_questions = next
            .outstanding_questions
            .checked_sub(1)
            .ok_or(FrontierError::Invalid("external pending counter underflow"))?;
        let reported = match completion.result() {
            Ok(outcome) => {
                if outcome.report.strict_native_nodes.is_some() {
                    return Err(FrontierError::Invalid(
                        "foreign nodes cannot attest native credit",
                    ));
                }
                if outcome.report.readiness_barrier_completed {
                    next.reported_ready_samples = next
                        .reported_ready_samples
                        .checked_add(1)
                        .ok_or(FrontierError::Invalid("external ready counter overflow"))?;
                }
                outcome.report.raw.work.nodes
            }
            Err(_) => {
                next.failed_terminal_packets = next
                    .failed_terminal_packets
                    .checked_add(1)
                    .ok_or(FrontierError::Invalid("external failure counter overflow"))?;
                completion.attempt().and_then(|attempt| attempt.work.nodes)
            }
        };
        if let Some(nodes) = reported {
            next.reported_nodes_known_sum = next
                .reported_nodes_known_sum
                .checked_add(nodes)
                .ok_or(FrontierError::Invalid(
                    "external reported-node sum overflow",
                ))?;
            next.reported_nodes_known_samples = next
                .reported_nodes_known_samples
                .checked_add(1)
                .ok_or(FrontierError::Invalid(
                    "external reported-node samples overflow",
                ))?;
        } else {
            next.reported_nodes_unknown_samples = next
                .reported_nodes_unknown_samples
                .checked_add(1)
                .ok_or(FrontierError::Invalid("external unknown samples overflow"))?;
        }
        self.external.rows[reservation.slot].completion = Some(Arc::downgrade(completion));
        self.external.work = next;
        Ok(FrontierExternalAcceptance::RecordedTerminal)
    }

    fn check_external_source(
        &self,
        source: &ProfiledExternalRootSource,
    ) -> Result<(), FrontierError> {
        let original = self
            .external_source
            .as_ref()
            .ok_or(FrontierError::MissingAcceptance)?;
        if !original.same_source(source)
            || source.authority() != self.registration.authority
            || source.purpose() != ExternalEnginePurpose::ResearchChecker
        {
            return Err(FrontierError::ForeignRoot);
        }
        Ok(())
    }

    fn check_external_dispatch(
        &self,
        source: &ProfiledExternalRootSource,
        dispatch: &ProfiledExternalDispatch,
    ) -> Result<(), FrontierError> {
        let question = dispatch.question();
        if question.id.authority != self.registration.authority
            || question.root_budget != self.registration.root_budget
            || question.original_deadline != self.registration.original_deadline
            || question.recipient != SearchRecipient::Researcher
            || dispatch.scope().owner != source.worker_owner()
            || dispatch.process_identity() != source.process_identity()
            || match question.original_first_move {
                Some(first) => first
                    .decode()
                    .ok()
                    .and_then(|movement| BoardMove::try_from(movement).ok())
                    .is_none_or(|movement| {
                        !self.registration.root_moves.allowed().contains(&movement)
                    }),
                None => {
                    question.anchor_root_ply != ChessPly(0)
                        || question.anchor != self.registration.root_rules.snapshot().identity()
                }
            }
        {
            return Err(FrontierError::ForeignRoot);
        }
        Ok(())
    }
}
