//! One-shot policy evidence from the original V3 teacher and live root.
//!
//! This authority is neither a Rules proof nor a corpus row, label, or
//! physical shutdown receipt. A consumer must still join the exact native
//! submitted input to its accepted retirement and perform target validation.

use super::*;
use crate::contracts::SessionScope;
use rz_contracts::pals::Role;
use rz_runtime::contracts::ContractClock;
use std::sync::TryLockError;
use std::sync::atomic::Ordering;
use std::time::Instant;

/// Retains the original V3 root and the actual worker-issued objects.
pub struct RoventTeacherPolicyEvidenceV3 {
    admission: RoventCollectionRootAdmissionV3,
    dispatch: TeacherDispatch,
    completion: TeacherCompletion,
}

impl RoventTeacherPolicyEvidenceV3 {
    pub fn admission(&self) -> &RoventCollectionRootAdmissionV3 {
        &self.admission
    }
    pub fn dispatch(&self) -> &TeacherDispatch {
        &self.dispatch
    }
    pub fn completion(&self) -> &TeacherCompletion {
        &self.completion
    }
    pub fn into_originals(self) -> (TeacherDispatch, TeacherCompletion) {
        (self.dispatch, self.completion)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoventTeacherPolicySealFailureKindV3 {
    Busy,
    Cancelled,
    Expired,
    Revoked,
    Poisoned,
    Invalid,
    /// A panic may have crossed the one-shot claim. Retain this custody.
    Uncertain,
}

pub struct RoventTeacherPolicySealErrorV3 {
    pub kind: RoventTeacherPolicySealFailureKindV3,
    pub error: CheckerError,
}

/// Keep original objects in the caller's custody on every rejection. If an
/// unwind crosses the one-shot claim, its state remains unknown and must not
/// be retried or dropped as if it were unused.
pub struct RoventTeacherPolicyCustodyV3 {
    dispatch: Option<TeacherDispatch>,
    completion: Option<TeacherCompletion>,
    uncertain: bool,
    panic_payload: Option<Box<dyn std::any::Any + Send>>,
}

impl RoventTeacherPolicyCustodyV3 {
    pub fn new(dispatch: TeacherDispatch, completion: TeacherCompletion) -> Self {
        Self {
            dispatch: Some(dispatch),
            completion: Some(completion),
            uncertain: false,
            panic_payload: None,
        }
    }
    pub fn is_uncertain(&self) -> bool {
        self.uncertain
    }
    /// The payload remains with this same custody for the recovery owner.
    pub fn panic_payload(&self) -> Option<&(dyn std::any::Any + Send)> {
        self.panic_payload.as_deref()
    }
    pub fn into_originals(mut self) -> Result<(TeacherDispatch, TeacherCompletion), Self> {
        if self.uncertain || self.dispatch.is_none() || self.completion.is_none() {
            return Err(self);
        }
        Ok((
            self.dispatch.take().expect("checked dispatch"),
            self.completion.take().expect("checked completion"),
        ))
    }
}

impl Drop for RoventTeacherPolicyCustodyV3 {
    fn drop(&mut self) {
        if self.uncertain {
            quarantine_v3::mark_custody_violation();
            if let Some(original) = self.dispatch.take() {
                std::mem::forget(original);
            }
            if let Some(original) = self.completion.take() {
                std::mem::forget(original);
            }
            if let Some(payload) = self.panic_payload.take() {
                std::mem::forget(payload);
            }
        }
    }
}

impl RoventCollectionTeacherLifecycleV3 {
    /// Check the independently reconstructed Rules query and actual physical
    /// result before locking the original root. Under that same root lock,
    /// re-check scope, cancellation and deadlines and claim the exact worker
    /// completion once. No worker or filesystem operation runs under the lock.
    pub fn seal_policy_custody_v3(
        &mut self,
        custody: &mut RoventTeacherPolicyCustodyV3,
        expected_query: &rz_position::PositionSnapshot,
        expected_root_moves: Option<&[rz_position::BoardMove]>,
    ) -> Result<RoventTeacherPolicyEvidenceV3, RoventTeacherPolicySealErrorV3> {
        if custody.uncertain {
            return Err(RoventTeacherPolicySealErrorV3 {
                kind: RoventTeacherPolicySealFailureKindV3::Invalid,
                error: invalid("teacher /3 custody claim state unknown"),
            });
        }
        custody.uncertain = true;
        let (Some(dispatch), Some(completion)) = (&custody.dispatch, &custody.completion) else {
            custody.uncertain = false;
            return Err(RoventTeacherPolicySealErrorV3 {
                kind: RoventTeacherPolicySealFailureKindV3::Invalid,
                error: invalid("teacher /3 custody already consumed"),
            });
        };
        let mut kind = RoventTeacherPolicySealFailureKindV3::Invalid;
        let validation = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
            || -> Result<_, CheckerError> {
                if self.shutdown_requested {
                    kind = RoventTeacherPolicySealFailureKindV3::Cancelled;
                    return Err(invalid("teacher /3 lifecycle already shutting down"));
                }
                if !dispatch.original_position().same_state(expected_query)
                    || dispatch.original_root_moves() != expected_root_moves
                {
                    return Err(invalid(
                        "teacher /3 original Rules query or restriction differs",
                    ));
                }
                let owner = self
                    .owner
                    .as_ref()
                    .ok_or(invalid("teacher /3 owner absent"))?;
                let startup = owner
                    .verified_startup()?
                    .ok_or(invalid("teacher /3 startup absent"))?;
                // This check queries the live root, so it must run before taking
                // the root's current-authority lock below.
                owner.validate_completion(completion, dispatch)?;
                let registration = owner.worker.registration();
                let admission = owner.admission();
                let root = admission.root_admission();
                let context = root.original_context();
                let current = context.current.try_lock().map_err(|error| match error {
                    TryLockError::WouldBlock => {
                        kind = RoventTeacherPolicySealFailureKindV3::Busy;
                        invalid("teacher /3 authority busy")
                    }
                    TryLockError::Poisoned(_) => {
                        kind = RoventTeacherPolicySealFailureKindV3::Poisoned;
                        invalid("teacher /3 authority poisoned")
                    }
                })?;
                let clock = root.runtime_clock();
                let now = clock
                    .try_now()
                    .map_err(|_| invalid("teacher /3 clock unavailable"))?;
                if root.hard_deadline().clock != clock.domain()
                    || dispatch.original_deadline().clock != clock.domain()
                {
                    return Err(invalid("teacher /3 deadline clock domain mismatch"));
                }
                if *current != SessionScope::Search(context.authority) {
                    kind = RoventTeacherPolicySealFailureKindV3::Revoked;
                    return Err(invalid("teacher /3 policy root revoked"));
                }
                if context.cancellation.is_canceled()
                    || context.control.cancellation.load(Ordering::Acquire)
                    || completion.controls().is_cancelled()
                {
                    kind = RoventTeacherPolicySealFailureKindV3::Cancelled;
                    return Err(invalid("teacher /3 policy cancelled"));
                }
                if Instant::now() >= context.control.deadline
                    || root.hard_deadline().accepts(clock.domain(), now).is_err()
                    || dispatch
                        .original_deadline()
                        .accepts(clock.domain(), now)
                        .is_err()
                {
                    kind = RoventTeacherPolicySealFailureKindV3::Expired;
                    return Err(invalid("teacher /3 policy deadline unavailable"));
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
                    return Err(invalid(
                        "teacher /3 policy authority unavailable or incomplete",
                    ));
                }
                let now = clock
                    .try_now()
                    .map_err(|_| invalid("teacher /3 clock unavailable"))?;
                if context.cancellation.is_canceled()
                    || context.control.cancellation.load(Ordering::Acquire)
                    || completion.controls().is_cancelled()
                {
                    kind = RoventTeacherPolicySealFailureKindV3::Cancelled;
                    return Err(invalid("teacher /3 policy cancelled during validation"));
                }
                if Instant::now() >= context.control.deadline
                    || root.hard_deadline().accepts(clock.domain(), now).is_err()
                    || dispatch
                        .original_deadline()
                        .accepts(clock.domain(), now)
                        .is_err()
                {
                    kind = RoventTeacherPolicySealFailureKindV3::Expired;
                    return Err(invalid("teacher /3 policy expired during validation"));
                }
                let original = admission.clone();
                if !completion.try_claim_policy_once(dispatch) {
                    return Err(invalid("teacher /3 policy completion already claimed"));
                }
                Ok(original)
            },
        ));
        match validation {
            Ok(Ok(admission)) => {
                let dispatch = custody.dispatch.take().expect("validated dispatch custody");
                let completion = custody
                    .completion
                    .take()
                    .expect("validated completion custody");
                custody.uncertain = false;
                Ok(RoventTeacherPolicyEvidenceV3 {
                    admission,
                    dispatch,
                    completion,
                })
            }
            Ok(Err(error)) => {
                custody.uncertain = false;
                Err(RoventTeacherPolicySealErrorV3 { kind, error })
            }
            Err(payload) => {
                // A panic may have happened after the one-shot CAS. The caller
                // receives the same uncertain custody and original payload.
                custody.panic_payload = Some(payload);
                Err(RoventTeacherPolicySealErrorV3 {
                    kind: RoventTeacherPolicySealFailureKindV3::Uncertain,
                    error: invalid("teacher /3 policy claim panicked; original custody retained"),
                })
            }
        }
    }
}
