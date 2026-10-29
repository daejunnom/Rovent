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
pub struct RoventTeacherPolicySealFailureV2 {
    pub error: CheckerError,
    pub dispatch: TeacherDispatch,
    pub completion: TeacherCompletion,
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
        let validation = (|| -> Result<_, CheckerError> {
            if self.shutdown_requested {
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
                TryLockError::WouldBlock => invalid("teacher /2 authority busy"),
                TryLockError::Poisoned(_) => invalid("teacher /2 authority poisoned"),
            })?;
            let clock = root.runtime_clock();
            let now = clock.try_now().map_err(|_| invalid("teacher /2 clock unavailable"))?;
            if *current != SessionScope::Search(context.authority)
                || context.cancellation.is_canceled()
                || context.control.cancellation.load(Ordering::Acquire)
                || Instant::now() >= context.control.deadline
                || root.hard_deadline().accepts(clock.domain(), now).is_err()
                || dispatch.original_deadline().accepts(clock.domain(), now).is_err()
                || completion.controls().is_cancelled()
                || !matches!(dispatch.input_scope().role, Role::Proposer | Role::Critic)
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
            owner.validate_completion_provenance(&completion, &dispatch)?;
            // Re-check atomic cancellations and both deadlines immediately at
            // issuance; no fallible operation follows the original token CAS.
            let now = clock.try_now().map_err(|_| invalid("teacher /2 clock unavailable"))?;
            if context.cancellation.is_canceled()
                || context.control.cancellation.load(Ordering::Acquire)
                || Instant::now() >= context.control.deadline
                || root.hard_deadline().accepts(clock.domain(), now).is_err()
                || dispatch.original_deadline().accepts(clock.domain(), now).is_err()
            {
                return Err(invalid("teacher /2 policy authority expired during validation"));
            }
            if !completion.try_claim_policy_once(&dispatch) {
                return Err(invalid("teacher /2 policy completion already claimed"));
            }
            Ok(admission.clone())
        })();
        match validation {
            Ok(admission) => Ok(RoventTeacherPolicyEvidenceV2 { admission, dispatch, completion }),
            Err(error) => Err(Box::new(RoventTeacherPolicySealFailureV2 { error, dispatch, completion })),
        }
    }
}
