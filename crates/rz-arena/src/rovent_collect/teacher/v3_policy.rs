//! Bind a completed teacher query to its exact V3 capsule and live owner.
//! This creates no training target, retired record, or publication authority.

use super::super::producer_v3::SubmittedTeacherQueryV3;
use super::super::{ArenaError, invalid};
use super::v3_dispatch::TeacherCompletedV3;
use super::v3_session::RoventCollectionTeacherSessionV3;
use rz_contracts::{Digest, pals::Role};
use rz_experiments::pals_data_v3::GenerationRoleV3;
use rz_uci::rovent_external_launch::{
    RoventTeacherPolicyCustodyV3, RoventTeacherPolicyEvidenceV3, RoventTeacherPolicySealErrorV3,
    VerifiedRoventExternalStartup,
};
use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::time::Duration;

pub(crate) struct RoventSealedTeacherPolicyV3 {
    capsule: Arc<SubmittedTeacherQueryV3>,
    startup: VerifiedRoventExternalStartup,
    actor_id: String,
    soft_quantum: Duration,
    evidence: RoventTeacherPolicyEvidenceV3,
}

impl RoventSealedTeacherPolicyV3 {
    pub(crate) fn capsule(&self) -> &Arc<SubmittedTeacherQueryV3> {
        &self.capsule
    }
    pub(crate) fn startup(&self) -> &VerifiedRoventExternalStartup {
        &self.startup
    }
    pub(crate) fn actor_id(&self) -> &str {
        &self.actor_id
    }
    pub(crate) fn soft_quantum(&self) -> Duration {
        self.soft_quantum
    }
    pub(crate) fn evidence(&self) -> &RoventTeacherPolicyEvidenceV3 {
        &self.evidence
    }
}

pub(crate) enum TeacherPolicySealCauseV3 {
    Validation(ArenaError),
    Panic(Box<dyn Any + Send>),
    Native(RoventTeacherPolicySealErrorV3),
}

/// Preserve all original evidence on errors, cancellation, and uncertain
/// one-shot claims. A post-seal cancellation retains the consumed evidence.
pub(crate) enum TeacherPolicySealRejectionV3 {
    BeforeClaim {
        cause: TeacherPolicySealCauseV3,
        completed: TeacherCompletedV3,
    },
    Custody {
        capsule: Arc<SubmittedTeacherQueryV3>,
        startup: VerifiedRoventExternalStartup,
        actor_id: String,
        soft_quantum: Duration,
        custody: RoventTeacherPolicyCustodyV3,
        cause: TeacherPolicySealCauseV3,
    },
    CancelledAfterSeal(RoventSealedTeacherPolicyV3),
}

pub(crate) fn seal_teacher_policy_v3(
    session: &mut RoventCollectionTeacherSessionV3,
    completed: TeacherCompletedV3,
) -> Result<RoventSealedTeacherPolicyV3, Box<TeacherPolicySealRejectionV3>> {
    let checked = catch_unwind(AssertUnwindSafe(|| -> Result<_, ArenaError> {
        let capsule = &completed.capsule;
        if completed.cancel_requested
            || completed.cancel_failure.is_some()
            || completed.validation_error.is_some()
            || capsule.is_canceled()
            || !capsule.has_policy_candidates()
            || session.is_closing()
            || !capsule
                .admission()
                .same_original_admission(session.original_admission())
        {
            return Err(invalid(
                "V3 teacher completion canceled, invalid, or root differs",
            ));
        }
        let input = capsule.input_seal();
        let admission = session.original_admission();
        let lock = admission.native_origin().original_input_lock();
        if input.teacher_actor_id.as_deref() != Some(completed.actor_id.as_str())
            || input.manifest_sha256 != session.manifest_lock().sha256(lock)?
        {
            return Err(invalid("V3 teacher completion actor or manifest differs"));
        }
        admission
            .validate_ticket(&input.ticket)
            .map_err(|e| invalid(e.to_string()))?;
        let role = match input.role {
            GenerationRoleV3::Proposer => Role::Proposer,
            GenerationRoleV3::Critic => Role::Critic,
            GenerationRoleV3::Validator => return Err(invalid("V3 teacher policy requires P/C")),
        };
        let parse = |value: &str| {
            rz_eval::asset::parse_sha256(value)
                .map(Digest)
                .map_err(|e| invalid(e.to_string()))
        };
        let registration = session
            .actor_registration(&completed.actor_id)
            .ok_or_else(|| invalid("V3 teacher actor registration absent"))?;
        let pin = session
            .registration()
            .teachers
            .get(&completed.actor_id)
            .ok_or_else(|| invalid("V3 teacher actor pin absent"))?;
        let scope = completed.dispatch.input_scope();
        let root = admission.root_admission();
        if input.ticket.role != role
            || scope.authority != input.ticket.authority
            || scope.role != role
            || scope.input_sha256 != parse(&input.input_sha256)?
            || scope.view_sha256 != parse(&input.selected_view_sha256)?
            || scope.source_sha256 != parse(&session.registration().source_sha256)?
            || scope.rules_semantic != parse(&input.anchor.rules_state_sha256)?
            || scope.input_lock != root.startup_lock_digest()
            || scope.root_admission != root.descriptor().digest()?
            || scope.root_budget != admission.root_budget()
            || scope.frozen_actor != registration.frozen_actor
            || completed.dispatch.id().collection_owner != registration.collection_owner
            || completed.dispatch.original_deadline() != input.ticket.original_deadline
            || completed.dispatch.original_root_moves().is_some()
            || completed.startup.process_identity() != completed.dispatch.process_identity()
            || completed.startup.canonical_sha256() != pin.canonical_sha256
            || completed.startup.capability_sha256() != pin.capability_sha256
            || completed.startup.input_lock_sha256() != super::super::hex(registration.input_lock)
        {
            return Err(invalid("V3 teacher completion differs from sealed query"));
        }
        let query = capsule.original_teacher_query_position()?.snapshot();
        if capsule.is_canceled() {
            return Err(invalid("V3 teacher capsule canceled before claim"));
        }
        Ok(query)
    }));
    let query = match checked {
        Ok(Ok(query)) => query,
        Ok(Err(error)) => {
            return Err(Box::new(TeacherPolicySealRejectionV3::BeforeClaim {
                cause: TeacherPolicySealCauseV3::Validation(error),
                completed,
            }));
        }
        Err(payload) => {
            return Err(Box::new(TeacherPolicySealRejectionV3::BeforeClaim {
                cause: TeacherPolicySealCauseV3::Panic(payload),
                completed,
            }));
        }
    };
    let Some(lifecycle) = session.lifecycle_mut(&completed.actor_id) else {
        return Err(Box::new(TeacherPolicySealRejectionV3::BeforeClaim {
            cause: TeacherPolicySealCauseV3::Validation(invalid("V3 teacher lifecycle absent")),
            completed,
        }));
    };
    let TeacherCompletedV3 {
        capsule,
        startup,
        actor_id,
        soft_quantum,
        dispatch,
        completion,
        cancel_requested: _,
        cancel_failure: _,
        validation_error: _,
    } = completed;
    let mut custody = RoventTeacherPolicyCustodyV3::new(dispatch, completion);
    let result = catch_unwind(AssertUnwindSafe(|| {
        lifecycle.seal_policy_custody_v3(&mut custody, &query, None)
    }));
    match result {
        Ok(Ok(evidence)) => {
            let sealed = RoventSealedTeacherPolicyV3 {
                capsule,
                startup,
                actor_id,
                soft_quantum,
                evidence,
            };
            if sealed.capsule.is_canceled() {
                Err(Box::new(TeacherPolicySealRejectionV3::CancelledAfterSeal(
                    sealed,
                )))
            } else {
                Ok(sealed)
            }
        }
        Ok(Err(error)) => Err(Box::new(TeacherPolicySealRejectionV3::Custody {
            capsule,
            startup,
            actor_id,
            soft_quantum,
            custody,
            cause: TeacherPolicySealCauseV3::Native(error),
        })),
        Err(payload) => Err(Box::new(TeacherPolicySealRejectionV3::Custody {
            capsule,
            startup,
            actor_id,
            soft_quantum,
            custody,
            cause: TeacherPolicySealCauseV3::Panic(payload),
        })),
    }
}
