//! Linearized, one-shot provenance for a completed V2 teacher opinion.
//!
//! This is an authority to inspect the original completed opinion, not a
//! Rules proof, WDL target, completed collection or publication authority.
use super::*;
use crate::contracts::SessionScope;
use rz_contracts::pals::Role;
use rz_runtime::contracts::ContractClock;
use std::sync::TryLockError;

/// Non-cloneable original completion bound to the actual root owner.
pub struct RoventTeacherPolicyEvidenceV2 {
    admission: crate::pals_adaptive::RoventCollectionRootAdmissionV2,
    dispatch: TeacherDispatch,
    completion: TeacherCompletion,
}

impl RoventTeacherPolicyEvidenceV2 {
    pub fn admission(&self) -> &crate::pals_adaptive::RoventCollectionRootAdmissionV2 {
        &self.admission
    }
    pub fn dispatch(&self) -> &TeacherDispatch {
        &self.dispatch
    }
    pub fn completion(&self) -> &TeacherCompletion {
        &self.completion
    }
    /// Transfer, rather than reproduce, the worker-issued dispatch and result.
    pub fn into_originals(self) -> (TeacherDispatch, TeacherCompletion) {
        (self.dispatch, self.completion)
    }
}

/// Rejection preserves the exact caller-owned dispatch and completion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoventTeacherPolicySealFailureKindV2 {
    Busy,
    Cancelled,
    Expired,
    Revoked,
    Poisoned,
    Invalid,
}

pub struct RoventTeacherPolicySealFailureV2 {
    pub kind: RoventTeacherPolicySealFailureKindV2,
    pub error: CheckerError,
    pub dispatch: TeacherDispatch,
    pub completion: TeacherCompletion,
}

/// Original worker objects stay with the caller across an unwind. After an
/// unwind the caller must quarantine this custody, not infer CAS state.
pub struct RoventTeacherPolicyCustodyV2 {
    dispatch: Option<TeacherDispatch>,
    completion: Option<TeacherCompletion>,
    uncertain: bool,
}
impl RoventTeacherPolicyCustodyV2 {
    pub fn new(dispatch: TeacherDispatch, completion: TeacherCompletion) -> Self {
        Self { dispatch: Some(dispatch), completion: Some(completion), uncertain: false }
    }
    pub fn is_uncertain(&self) -> bool { self.uncertain }
    pub fn into_originals(mut self) -> Result<(TeacherDispatch, TeacherCompletion), Self> {
        if self.uncertain || self.dispatch.is_none() || self.completion.is_none() {
            return Err(self);
        }
        Ok((self.dispatch.take().expect("checked dispatch"), self.completion.take().expect("checked completion")))
    }
}
impl Drop for RoventTeacherPolicyCustodyV2 {
    fn drop(&mut self) {
        if self.uncertain {
            // A panic may have occurred on either side of the one-shot CAS.
            // Preserve any original objects rather than silently release an
            // unknown claim. The caller may instead retain the whole custody.
            quarantine_v2::mark_custody_violation();
            if let Some(dispatch) = self.dispatch.take() { std::mem::forget(dispatch); }
            if let Some(completion) = self.completion.take() { std::mem::forget(completion); }
        }
    }
}

pub struct RoventTeacherPolicySealErrorV2 {
    pub kind: RoventTeacherPolicySealFailureKindV2,
    pub error: CheckerError,
}

impl RoventCollectionTeacherLifecycleV2 {
    /// Validate and claim one actual completion at the original UCI root's
    /// authority linearization point. No external process or producer lock is
    /// touched while the guard is held. Cancellation, deadline, root change,
    /// failed result or missing physical evidence never consume the claim.
    pub fn seal_policy_completion(
        &mut self,
        dispatch: TeacherDispatch,
        completion: TeacherCompletion,
        expected_query: &rz_position::PositionSnapshot,
        expected_root_moves: Option<&[rz_position::BoardMove]>,
    ) -> Result<RoventTeacherPolicyEvidenceV2, Box<RoventTeacherPolicySealFailureV2>> {
        let mut custody = RoventTeacherPolicyCustodyV2::new(dispatch, completion);
        match self.seal_policy_custody(&mut custody, expected_query, expected_root_moves) {
            Ok(evidence) => Ok(evidence),
            Err(failure) => {
                let (dispatch, completion) = match custody.into_originals() {
                    Ok(originals) => originals,
                    Err(custody) => {
                        // An impossible partial custody is not safe to drop.
                        std::mem::forget(custody);
                        panic!("failure lost teacher custody invariant");
                    }
                };
                Err(Box::new(RoventTeacherPolicySealFailureV2 { kind: failure.kind, error: failure.error, dispatch, completion }))
            }
        }
    }

    /// Keep original objects in caller custody until validation and the claim
    /// complete. Do not retry after unwinding: claim state may be unknown.
    pub fn seal_policy_custody(
        &mut self,
        custody: &mut RoventTeacherPolicyCustodyV2,
        expected_query: &rz_position::PositionSnapshot,
        expected_root_moves: Option<&[rz_position::BoardMove]>,
    ) -> Result<RoventTeacherPolicyEvidenceV2, RoventTeacherPolicySealErrorV2> {
        if custody.uncertain {
            return Err(RoventTeacherPolicySealErrorV2 { kind: RoventTeacherPolicySealFailureKindV2::Invalid, error: invalid("teacher /2 custody claim state unknown") });
        }
        custody.uncertain = true;
        let (Some(dispatch), Some(completion)) = (&custody.dispatch, &custody.completion) else {
            custody.uncertain = false;
            return Err(RoventTeacherPolicySealErrorV2 { kind: RoventTeacherPolicySealFailureKindV2::Invalid, error: invalid("teacher /2 custody already consumed") });
        };
        let mut kind = RoventTeacherPolicySealFailureKindV2::Invalid;
        let validation = (|| -> Result<_, CheckerError> {
            if self.shutdown_requested {
                kind = RoventTeacherPolicySealFailureKindV2::Cancelled;
                return Err(invalid("teacher /2 lifecycle already shutting down"));
            }
            // These immutable snapshots carry the actual Rules history. A
            // matching caller-supplied digest alone is never sufficient.
            // The caller must reconstruct expected_query from its original
            // retired row, not from the teacher dispatch itself.
            if !dispatch.original_position().same_state(expected_query)
                || dispatch.original_root_moves() != expected_root_moves
            {
                return Err(invalid("teacher /2 original Rules query or restriction differs"));
            }
            let owner = self.owner.as_ref().ok_or(invalid("teacher /2 owner absent"))?;
            // Startup is read before acquiring the UCI authority guard: the
            // worker owns its startup mutex and must not be waited on inside
            // the root's linearization boundary.
            let startup = owner.verified_startup()?
                .ok_or(invalid("teacher /2 original worker startup absent"))?;
            let registration = owner.worker.registration();
            let admission = owner.admission();
            let root = admission.root_admission();
            let context = root.original_context();
            let current = context.current.try_lock().map_err(|error| match error {
                TryLockError::WouldBlock => {
                    kind = RoventTeacherPolicySealFailureKindV2::Busy;
                    invalid("teacher /2 authority busy")
                }
                TryLockError::Poisoned(_) => {
                    kind = RoventTeacherPolicySealFailureKindV2::Poisoned;
                    invalid("teacher /2 authority poisoned")
                }
            })?;
            let clock = root.runtime_clock();
            let now = clock.try_now().map_err(|_| invalid("teacher /2 clock unavailable"))?;
            if root.hard_deadline().clock != clock.domain()
                || dispatch.original_deadline().clock != clock.domain()
            {
                return Err(invalid("teacher /2 deadline clock domain mismatch"));
            }
            if *current != SessionScope::Search(context.authority) {
                kind = RoventTeacherPolicySealFailureKindV2::Revoked;
                return Err(invalid("teacher /2 policy root revoked"));
            }
            if context.cancellation.is_canceled()
                || context.control.cancellation.load(Ordering::Acquire)
                || completion.controls().is_cancelled()
            {
                kind = RoventTeacherPolicySealFailureKindV2::Cancelled;
                return Err(invalid("teacher /2 policy cancelled"));
            }
            if Instant::now() >= context.control.deadline
                || root.hard_deadline().accepts(clock.domain(), now).is_err()
                || dispatch.original_deadline().accepts(clock.domain(), now).is_err()
            {
                kind = RoventTeacherPolicySealFailureKindV2::Expired;
                return Err(invalid("teacher /2 policy deadline unavailable"));
            }
            if !matches!(dispatch.input_scope().role, Role::Proposer | Role::Critic)
                || completion.result().is_err()
                || dispatch.process_identity() != startup.process_identity()
                || dispatch.id().collection_owner != registration.collection_owner
                || dispatch.input_scope().authority != registration.authority
                || dispatch.input_scope().input_lock != registration.input_lock
                || dispatch.input_scope().root_admission != registration.root_admission
                || dispatch.input_scope().root_budget != registration.root_budget
                || dispatch.input_scope().frozen_actor != registration.frozen_actor
                || dispatch.original_deadline().clock != registration.original_deadline.clock
                || dispatch.original_deadline().at > registration.original_deadline.at
            {
                return Err(invalid("teacher /2 policy authority unavailable or incomplete"));
            }
            owner.validate_completion_provenance(completion, dispatch)?;
            // Re-check atomic cancellations and both deadlines immediately at
            // issuance; no fallible operation follows the original token CAS.
            let now = clock.try_now().map_err(|_| invalid("teacher /2 clock unavailable"))?;
            if context.cancellation.is_canceled()
                || context.control.cancellation.load(Ordering::Acquire)
                || completion.controls().is_cancelled()
            {
                kind = RoventTeacherPolicySealFailureKindV2::Cancelled;
                return Err(invalid("teacher /2 policy cancelled during validation"));
            }
            if Instant::now() >= context.control.deadline
                || root.hard_deadline().accepts(clock.domain(), now).is_err()
                || dispatch.original_deadline().accepts(clock.domain(), now).is_err()
            {
                kind = RoventTeacherPolicySealFailureKindV2::Expired;
                return Err(invalid("teacher /2 policy authority expired during validation"));
            }
            let accepted_admission = admission.clone();
            if !completion.try_claim_policy_once(dispatch) {
                return Err(invalid("teacher /2 policy completion already claimed"));
            }
            Ok(accepted_admission)
        })();
        match validation {
            Ok(admission) => {
                // Both options were inspected above and cannot be externally
                // mutated while borrowed. No callback runs between claim and
                // these ownership transfers.
                let dispatch = custody.dispatch.take().expect("validated dispatch custody");
                let completion = custody.completion.take().expect("validated completion custody");
                custody.uncertain = false;
                Ok(RoventTeacherPolicyEvidenceV2 { admission, dispatch, completion })
            }
            Err(error) => {
                custody.uncertain = false;
                Err(RoventTeacherPolicySealErrorV2 { kind, error })
            },
        }
    }
}
