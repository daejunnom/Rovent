//! Compute a conditional policy from the original V3 completion and retired
//! row. This preparation does not alter a producer, publish a row, or imply
//! training eligibility. The original one-shot evidence remains caller-owned.

use super::collection::{ROVENT_COLLECTION_TEACHER_BINDING, scope_json};
use super::v3_policy::RoventSealedTeacherPolicyV3;
use super::*;
use crate::rovent_collect::producer_v3::RoventRetiredLockV3Record;

pub(crate) struct PreparedTeacherPolicyLockV3 {
    sealed: RoventSealedTeacherPolicyV3,
    ticket: RoleTicketEnvelope,
    target: GenerationPolicyTargetV3,
    source: RoventHeadSource,
}

impl PreparedTeacherPolicyLockV3 {
    pub(crate) fn sealed(&self) -> &RoventSealedTeacherPolicyV3 {
        &self.sealed
    }
    pub(crate) fn ticket(&self) -> RoleTicketEnvelope {
        self.ticket
    }
    pub(crate) fn target(&self) -> &GenerationPolicyTargetV3 {
        &self.target
    }
    pub(crate) fn source(&self) -> &RoventHeadSource {
        &self.source
    }
    pub(crate) fn into_sealed(self) -> RoventSealedTeacherPolicyV3 {
        self.sealed
    }
}

pub(crate) struct PrepareTeacherPolicyFailureV3 {
    pub error: ArenaError,
    pub panic: Option<Box<dyn std::any::Any + Send>>,
    pub policy_unavailable: bool,
}

/// The actor's actual one-shot policy evidence must already exist. A native
/// capture by itself, a raw completion, or a matching digest cannot enter.
pub(crate) fn prepare_teacher_policy_lock_v3_in_slots(
    row: &RoventRetiredLockV3Record,
    sealed_slot: &mut Option<RoventSealedTeacherPolicyV3>,
    prepared_slot: &mut Option<PreparedTeacherPolicyLockV3>,
) -> Result<(), Box<PrepareTeacherPolicyFailureV3>> {
    let unavailable = std::cell::Cell::new(false);
    let prepare = || -> Result<(GenerationPolicyTargetV3, RoventHeadSource), ArenaError> {
        check(prepared_slot.is_none(), "V3 teacher prepared slot occupied")?;
        let sealed = sealed_slot
            .as_ref()
            .ok_or_else(|| invalid("V3 teacher sealed slot empty"))?;
        let capsule = sealed.capsule();
        check(
            !capsule.is_ticket_canceled(),
            "V3 teacher input canceled before target preparation",
        )?;
        capsule.matches_original_row(row)?;
        let evidence = sealed.evidence();
        let dispatch = evidence.dispatch();
        let completion = evidence.completion();
        let input = capsule.input_seal();
        let record = &row.record().0;
        let admission = row.admission();
        let root = admission.root_admission();
        let registration = &row.manifest().registration;
        let pin = registration
            .teachers
            .get(sealed.actor_id())
            .ok_or_else(|| invalid("V3 teacher actor not registered in original manifest"))?
            .clone();
        let registered = RoventRegisteredTeacher {
            actor_id: sealed.actor_id().into(),
            pin: pin.clone(),
        };
        let parse = |value: &str| {
            rz_eval::asset::parse_sha256(value)
                .map(Digest)
                .map_err(|e| invalid(e.to_string()))
        };
        let position = capsule.original_teacher_query_position()?;
        let scope = dispatch.input_scope();
        check(
            evidence.admission().same_original_admission(admission)
                && input.ticket == row.ticket()
                && input.input_sha256 == record.input_sha256
                && input.selected_view_sha256 == record.input.view_sha256
                && input.anchor == record.input.rules_anchor
                && input.role == record.input.role
                && input.teacher_actor_id.as_deref() == Some(sealed.actor_id())
                && record.lineage.teacher_actor_id.as_deref() == Some(sealed.actor_id())
                && scope.authority == row.ticket().authority
                && scope.input_sha256 == parse(&input.input_sha256)?
                && scope.view_sha256 == parse(&input.selected_view_sha256)?
                && scope.source_sha256 == parse(&registration.source_sha256)?
                && scope.role == row.ticket().role
                && scope.rules_semantic == parse(&input.anchor.rules_state_sha256)?
                && scope.input_lock == root.startup_lock_digest()
                && scope.root_admission == root.descriptor().digest()?
                && scope.root_budget == admission.root_budget()
                && scope.frozen_actor == registered.frozen_actor_digest()?
                && dispatch
                    .original_position()
                    .same_state(&position.snapshot())
                && dispatch.original_root_moves().is_none()
                && dispatch.original_deadline() == row.ticket().original_deadline
                && dispatch.process_identity() == sealed.startup().process_identity()
                && sealed.startup().canonical_sha256() == pin.canonical_sha256
                && sealed.startup().capability_sha256() == pin.capability_sha256
                && sealed.startup().input_lock_sha256() == hex(root.startup_lock_digest())
                && completion.go_issue() == ExternalGoIssue::Issued
                && record.targets.policy.is_none()
                && !record.head_sources.contains_key("policy")
                && !record.input.legal_moves.is_empty(),
            "V3 sealed teacher evidence differs from original row/actor/root/position",
        )?;
        let outcome = completion
            .result()
            .as_ref()
            .map_err(|e| invalid(e.to_string()))?;
        let attempt = completion
            .attempt()
            .ok_or_else(|| invalid("V3 teacher physical attempt missing"))?;
        let binding = json!({"schema":ROVENT_COLLECTION_TEACHER_BINDING,
            "job_id":{"collection_owner":dispatch.id().collection_owner,"sequence":dispatch.id().sequence},
            "input_scope":scope_json(scope),"requested":dispatch.constraint(),
            "process":process_json(dispatch.process_identity()),
            "root_restriction":hex(dispatch.root_restriction()),"position_anchor":input.anchor,
            "root_moves":null,"soft_quantum_ns":sealed.soft_quantum().as_nanos().min(u128::from(u64::MAX)) as u64,
            "original_deadline":{"clock_epoch":dispatch.original_deadline().clock.0.0,"at_ns":dispatch.original_deadline().at.0},
            "root_descriptor":root.descriptor(),"root_descriptor_sha256":hex(root.descriptor().digest()?)});
        let projected = record.input.common_validation_projection();
        let (ranked, depth) = validate_completed_teacher_opinions_classified(
            &projected,
            &pin,
            outcome,
            &unavailable,
        )?;
        prepare_completed_teacher_policy(
            &projected,
            &record.input_sha256,
            record.native.retired_sequence,
            sealed.actor_id().into(),
            pin,
            binding,
            sealed.startup(),
            outcome,
            attempt,
            completion.timing(),
            ranked,
            depth,
        )
    };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(prepare)) {
        Ok(Ok((target, source))) => {
            let sealed = sealed_slot.take().expect("validated original sealed slot");
            *prepared_slot = Some(PreparedTeacherPolicyLockV3 {
                sealed,
                ticket: row.ticket(),
                target,
                source,
            });
            Ok(())
        }
        Ok(Err(error)) => Err(Box::new(PrepareTeacherPolicyFailureV3 {
            error,
            panic: None,
            policy_unavailable: unavailable.get(),
        })),
        Err(panic) => Err(Box::new(PrepareTeacherPolicyFailureV3 {
            error: invalid("V3 teacher policy preparation panicked"),
            panic: Some(panic),
            policy_unavailable: false,
        })),
    }
}
