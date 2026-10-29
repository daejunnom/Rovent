//! Move-only V2 teacher evidence is checked and transformed outside the
//! producer lock. A later short producer commit only transfers the prepared
//! target to the same original retired row.
use super::collection::{scope_json, ROVENT_COLLECTION_TEACHER_BINDING};
use super::v2_session::SealedTeacherPolicyV2;
use super::*;

/// Carries the original one-shot evidence until the caller explicitly takes
/// it back. The prepared target is not a published or training-ready record.
pub(crate) struct PreparedTeacherPolicyLockV2 {
    sealed: SealedTeacherPolicyV2,
    ticket: RoleTicketEnvelope,
    target: GenerationPolicyTargetV3,
    source: RoventHeadSource,
}

pub(crate) struct PrepareTeacherPolicyFailureV2 {
    pub(crate) error: ArenaError,
    pub(crate) sealed: SealedTeacherPolicyV2,
    pub(crate) panic: Option<Box<dyn std::any::Any + Send>>,
}

pub(crate) struct CommitTeacherPolicyFailureV2 {
    pub(crate) error: ArenaError,
    pub(crate) panic: Option<Box<dyn std::any::Any + Send>>,
}

/// Validate original row, teacher and Rules identity and calculate the
/// policy outside the observer/producer mutex. Errors return the original
/// non-cloneable evidence for recovery. No root/worker is called here.
pub(crate) fn prepare_teacher_policy_lock_v2(
    row: &RoventRetiredLockV2Record,
    sealed: SealedTeacherPolicyV2,
) -> Result<PreparedTeacherPolicyLockV2, Box<PrepareTeacherPolicyFailureV2>> {
    let prepare = || -> Result<(GenerationPolicyTargetV3, RoventHeadSource), ArenaError> {
        let evidence = &sealed.evidence;
        let admission = evidence.admission();
        let dispatch = evidence.dispatch();
        let completion = evidence.completion();
        let input = &sealed.input;
        let row_input = &row.record.0.input;
        let root = row.admission.root_admission();
        let scope = dispatch.input_scope();
        let registration = &row.manifest_owner.registration;
        let parse_digest = |value: &str| {
            rz_eval::asset::parse_sha256(value)
                .map(Digest)
                .map_err(|error| invalid(error.to_string()))
        };
        let position = row.original_teacher_query_position()?;
        check(
            admission.same_original_admission(&row.admission)
                && row.matches_manifest_owner(&sealed.manifest_owner)
                && input.manifest_sha256 == row.manifest_sha256
                && input.ticket == row.ticket
                && input.input_sha256 == row.record.0.input_sha256
                && input.selected_view_sha256 == row_input.view_sha256
                && input.anchor == row_input.rules_anchor
                && input.role == row_input.role
                && input.teacher_actor_id.as_deref() == Some(sealed.actor_id.as_str())
                && row.record.0.lineage.teacher_actor_id.as_deref()
                    == Some(sealed.actor_id.as_str())
                && sealed.registered.actor_id == sealed.actor_id
                && registration.teachers.get(&sealed.actor_id) == Some(&sealed.registered.pin)
                && scope.authority == row.ticket.authority
                && scope.input_sha256 == parse_digest(&input.input_sha256)?
                && scope.view_sha256 == parse_digest(&input.selected_view_sha256)?
                && scope.source_sha256 == parse_digest(&registration.source_sha256)?
                && scope.role == row.ticket.role
                && scope.rules_semantic == parse_digest(&input.anchor.rules_state_sha256)?
                && scope.input_lock == root.startup_lock_digest()
                && scope.root_admission == root.descriptor().digest()?
                && scope.root_budget == row.admission.root_budget()
                && scope.frozen_actor == sealed.registered.frozen_actor_digest()?
                && dispatch
                    .original_position()
                    .same_state(&position.snapshot())
                && dispatch.original_root_moves().is_none()
                && dispatch.original_deadline() == row.ticket.original_deadline
                && dispatch.process_identity() == sealed.startup.process_identity()
                && sealed.startup.canonical_sha256() == sealed.registered.pin.canonical_sha256
                && sealed.startup.capability_sha256() == sealed.registered.pin.capability_sha256
                && sealed.startup.input_lock_sha256() == hex(root.startup_lock_digest())
                && completion.go_issue() == ExternalGoIssue::Issued
                && row.record.0.targets.policy.is_none()
                && !row.record.0.head_sources.contains_key("policy")
                && !row_input.legal_moves.is_empty(),
            "lock-v2 sealed teacher policy differs from original row/actor/root/position",
        )?;
        let outcome = completion
            .result()
            .as_ref()
            .map_err(|error| invalid(error.to_string()))?;
        let attempt = completion
            .attempt()
            .ok_or_else(|| invalid("lock-v2 original teacher attempt missing"))?;
        let binding = json!({"schema":ROVENT_COLLECTION_TEACHER_BINDING,
            "job_id":{"collection_owner":dispatch.id().collection_owner,"sequence":dispatch.id().sequence},
            "input_scope":scope_json(scope),"requested":dispatch.constraint(),
            "process":process_json(dispatch.process_identity()),
            "root_restriction":hex(dispatch.root_restriction()),"position_anchor":input.anchor,
            "root_moves":null,"soft_quantum_ns":sealed.soft_quantum.as_nanos().min(u128::from(u64::MAX)) as u64,
            "original_deadline":{"clock_epoch":dispatch.original_deadline().clock.0.0,"at_ns":dispatch.original_deadline().at.0},
            "root_descriptor":root.descriptor(),"root_descriptor_sha256":hex(root.descriptor().digest()?)});
        // The original retirement sequence is used only inside this private,
        // unpublished calculation. The producer assigns the actual next
        // observation sequence at commit; neither temporary value is emitted.
        let projected_input = row_input.common_validation_projection();
        let (ranked, depth) =
            validate_completed_teacher_opinions(&projected_input, &sealed.registered.pin, outcome)?;
        prepare_completed_teacher_policy(
            &projected_input,
            &row.record.0.input_sha256,
            row.record.0.native.retired_sequence,
            sealed.actor_id.clone(),
            sealed.registered.pin.clone(),
            binding,
            &sealed.startup,
            outcome,
            attempt,
            completion.timing(),
            ranked,
            depth,
        )
    };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(prepare)) {
        Ok(Ok((target, source))) => Ok(PreparedTeacherPolicyLockV2 {
            sealed,
            ticket: row.ticket,
            target,
            source,
        }),
        Ok(Err(error)) => Err(Box::new(PrepareTeacherPolicyFailureV2 {
            error,
            sealed,
            panic: None,
        })),
        Err(panic) => Err(Box::new(PrepareTeacherPolicyFailureV2 {
            error: invalid("V2 teacher policy preparation panicked"),
            sealed,
            panic: Some(panic),
        })),
    }
}

impl RoventCorpusProducerLockV2 {
    /// Commit a prepared policy without observing the worker, root mutex or
    /// filesystem. The caller holds custody on every error and unwind.
    pub(crate) fn attach_prepared_teacher_policy_v2(
        &mut self,
        row: &mut RoventRetiredLockV2Record,
        custody: &mut Option<PreparedTeacherPolicyLockV2>,
    ) -> Result<SealedTeacherPolicyV2, Box<CommitTeacherPolicyFailureV2>> {
        let validate = || -> Result<(), ArenaError> {
            let prepared = custody
                .as_ref()
                .ok_or_else(|| invalid("V2 prepared policy custody is empty"))?;
            self.validate_teacher_row(row)?;
            check(
                prepared.ticket == row.ticket
                    && row.matches_manifest_owner(&prepared.sealed.manifest_owner)
                    && prepared.sealed.input.ticket == row.ticket
                    && prepared.sealed.input.input_sha256 == row.record.0.input_sha256
                    && prepared
                        .sealed
                        .evidence
                        .admission()
                        .same_original_admission(&row.admission)
                    && row.record.0.targets.policy.is_none()
                    && !row.record.0.head_sources.contains_key("policy"),
                "lock-v2 prepared policy no longer belongs to original row/producer",
            )
        };
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(validate)) {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                return Err(Box::new(CommitTeacherPolicyFailureV2 {
                    error,
                    panic: None,
                }))
            }
            Err(panic) => {
                return Err(Box::new(CommitTeacherPolicyFailureV2 {
                    error: invalid("V2 teacher policy commit validation panicked"),
                    panic: Some(panic),
                }))
            }
        }
        let producer = self.teacher_target_producer_mut();
        let Some(sequence) = producer.sequence.checked_add(1) else {
            return Err(Box::new(CommitTeacherPolicyFailureV2 {
                error: invalid("Rovent collector sequence exhausted"),
                panic: None,
            }));
        };
        // Build the replacement map before changing any row or sequence.
        // Recoverable unwinds return the original evidence unchanged.
        let sources = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut sources = row.record.0.head_sources.clone();
            let mut source = custody.as_ref().expect("validated custody").source.clone();
            source.observed_sequence = sequence;
            sources.insert("policy".into(), source);
            sources
        }));
        let sources = match sources {
            Ok(sources) => sources,
            Err(panic) => {
                return Err(Box::new(CommitTeacherPolicyFailureV2 {
                    error: invalid("V2 teacher policy commit preparation panicked"),
                    panic: Some(panic),
                }));
            }
        };
        let Some(mut prepared) = custody.take() else {
            return Err(Box::new(CommitTeacherPolicyFailureV2 {
                error: invalid("V2 prepared policy custody disappeared"),
                panic: None,
            }));
        };
        prepared.target.observed_sequence = sequence;
        producer.sequence = sequence;
        row.record.0.targets.policy = Some(prepared.target);
        row.record.0.head_sources = sources;
        Ok(prepared.sealed)
    }
}
