//! Experimental diagnostic projection from the original sealed teacher owner.
//! This borrows the original evidence; it neither creates a policy target nor
//! provides a new publication, retirement, or physical completion authority.
//! This is a repeatable diagnostic, not a linearized publication operation.
//! A retired-row-only seal has no retained ticket guard and cannot use this
//! path; a missing guard or expired original deadline fails closed. A guard
//! check observes cancellation but does not lock against a later cancellation.
use super::v2_session::SealedTeacherPolicyV2;
use super::*;
use rz_experiments::pals_data_v3::teacher_exact_rank::{
    TEACHER_EXACT_RANK_OBSERVATION_DOMAIN, TeacherExactGoIdentity, TeacherExactRankObservation,
    TeacherExactRankOpinion, validate_teacher_exact_rank_observation,
};
use rz_search::cpu_checker::ExternalBound;
use std::time::Instant;

/// Project only while the actual one-shot sealed owner and matching retired
/// input row are held. All fields come from those owners, never a decoded
/// request. Any unavailable time, witness, or original identity is an error.
pub(crate) fn project_original_exact_observation_v2(
    row: &RoventRetiredLockV2Record,
    sealed: &SealedTeacherPolicyV2,
) -> Result<TeacherExactRankObservation, ArenaError> {
    // A row-only seal does not retain a live ticket cancellation guard. Never
    // turn that missing authority into a clean cancellation observation.
    let guard = sealed
        .speculative_guard
        .as_ref()
        .ok_or_else(|| invalid("exact witness original ticket cancellation guard unavailable"))?;
    check(!guard.is_ticket_canceled(), "exact witness ticket canceled")?;
    guard.matches_original_row(row)?;
    let evidence = &sealed.evidence;
    let dispatch = evidence.dispatch();
    let completion = evidence.completion();
    let input = &sealed.input;
    let row_input = &row.record.0.input;
    let root = row.admission.root_admission();
    let scope = dispatch.input_scope();
    let registration = &row.manifest_owner.registration;
    let parse = |s: &str| {
        rz_eval::asset::parse_sha256(s)
            .map(Digest)
            .map_err(|e| invalid(e.to_string()))
    };
    let position = row.original_teacher_query_position()?;
    check(
        evidence.admission().same_original_admission(&row.admission)
            && row.matches_manifest_owner(&sealed.manifest_owner)
            && input.manifest_sha256 == row.manifest_sha256
            && input.ticket == row.ticket
            && input.input_sha256 == row.record.0.input_sha256
            && input.selected_view_sha256 == row_input.view_sha256
            && input.anchor == row_input.rules_anchor
            && input.role == row_input.role
            && input.teacher_actor_id.as_deref() == Some(sealed.actor_id.as_str())
            && row.record.0.lineage.teacher_actor_id.as_deref() == Some(sealed.actor_id.as_str())
            && sealed.registered.actor_id == sealed.actor_id
            && registration.teachers.get(&sealed.actor_id) == Some(&sealed.registered.pin)
            && scope.authority == row.ticket.authority
            && scope.input_sha256 == parse(&input.input_sha256)?
            && scope.view_sha256 == parse(&input.selected_view_sha256)?
            && scope.source_sha256 == parse(&registration.source_sha256)?
            && scope.role == row.ticket.role
            && scope.rules_semantic == parse(&input.anchor.rules_state_sha256)?
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
            && completion.matches_submission(dispatch)
            && completion.go_issue() == ExternalGoIssue::Issued,
        "exact witness differs from original row, owner, actor or query",
    )?;
    let outcome = completion
        .result()
        .as_ref()
        .map_err(|e| invalid(e.to_string()))?;
    let witness = completion
        .completed_exact_rank_set()
        .ok_or_else(|| invalid("exact witness unavailable"))?;
    check(
        witness.request_id() == outcome.report.raw.request_id,
        "exact witness request differs",
    )?;
    let origin = root.runtime_clock().origin();
    let tick = |value: Option<Instant>| -> Result<u64, ArenaError> {
        let nanos = value
            .ok_or_else(|| invalid("exact witness time unavailable"))?
            .checked_duration_since(origin)
            .ok_or_else(|| invalid("exact witness time predates original clock"))?
            .as_nanos();
        u64::try_from(nanos).map_err(|_| invalid("exact witness time overflow"))
    };
    let timing = completion.timing();
    let controls = completion.controls();
    let process = dispatch.process_identity();
    let identity = TeacherExactGoIdentity {
        input_sha256: input.input_sha256.clone(),
        view_sha256: input.selected_view_sha256.clone(),
        rules_state_sha256: input.anchor.rules_state_sha256.clone(),
        rules_history_sha256: input.anchor.rules_history_sha256.clone(),
        input_lock_sha256: hex(scope.input_lock),
        evaluation_namespace_sha256: hex(dispatch.evaluation_namespace()),
        root_restriction_sha256: hex(dispatch.root_restriction()),
        teacher_binary_sha256: sealed.registered.pin.binary_sha256.clone(),
        teacher_profile_sha256: sealed.registered.pin.profile_sha256.clone(),
        process_id: process.pid,
        process_group: process.process_group,
        process_start_ticks: process.proc_start_ticks,
        execution_epoch: scope.authority.epoch.0,
        clock_epoch: dispatch.original_deadline().clock.0.0,
        original_deadline_ns: dispatch.original_deadline().at.0,
        root_generation: scope.authority.root.0,
        collection_owner: dispatch.id().collection_owner,
        job_sequence: dispatch.id().sequence,
        request_id: witness.request_id(),
        requested_multipv: outcome.requested_multipv,
        root_moves: None,
    };
    let mut opinions = Vec::with_capacity(witness.opinions().len());
    for opinion in witness.opinions() {
        check(
            opinion.bound == ExternalBound::ExactReported,
            "exact witness bound differs",
        )?;
        // Recheck the whole PV against the independently reconstructed Rules
        // position. The worker also checked it on receipt of the original go.
        let mut current = position.clone();
        let mut pv = Vec::with_capacity(opinion.pv.len());
        for movement in &opinion.pv {
            let view = current.ordered_legal_moves();
            check(
                current
                    .play_status_from_view(&view)
                    .map_err(|e| invalid(e.to_string()))?
                    == rz_position::PlayStatus::Ongoing,
                "exact witness PV continues after terminal",
            )?;
            current
                .make_from_view(&view, *movement)
                .map_err(|e| invalid(e.to_string()))?;
            pv.push(
                Move16::pack(*movement)
                    .map_err(|e| invalid(e.to_string()))?
                    .bits(),
            );
        }
        opinions.push(TeacherExactRankOpinion {
            rank: opinion.rank,
            depth: opinion.depth,
            bound: "ExactReported".into(),
            pv,
        });
    }
    let observed = TeacherExactRankObservation {
        schema: TEACHER_EXACT_RANK_OBSERVATION_DOMAIN.into(),
        policy: witness.policy().into(),
        identity: identity.clone(),
        observation_sequence: witness.observation_sequence(),
        opinions,
        final_bestmove: Move16::pack(witness.final_bestmove())
            .map_err(|e| invalid(e.to_string()))?
            .bits(),
        final_bestmove_matches_rank1: witness.first_matches_final(),
        go_ns: tick(timing.go_issued_at)?,
        bestmove_ns: tick(timing.bestmove_drained_at)?,
        ready_ns: tick(timing.ready_completed_at)?,
        returned_ns: tick(timing.returned_at)?,
        cancel_sequence: controls.cancel_sequence,
        yield_sequence: controls.yield_sequence,
        stop_sent: timing.stop_sent_at.is_some(),
    };
    validate_teacher_exact_rank_observation(&observed, &identity, &row_input.legal_moves)
        .map_err(|e| invalid(e.to_string()))?;
    // A check is not a lock: cancellation immediately after this observation
    // is possible. A future publisher must linearize against the real owner.
    guard.matches_original_row(row)?;
    check(
        !guard.is_ticket_canceled(),
        "exact witness ticket canceled during projection",
    )?;
    let clock = root.runtime_clock();
    let now = clock.try_now().map_err(|e| invalid(e.to_string()))?;
    // RootAdmissionV2 binds its immutable runtime clock domain and origin to
    // this hard deadline during admission. Reuse that public original domain;
    // never construct or sample a fresh origin for these tick comparisons.
    let original_domain = root.hard_deadline().clock;
    check(
        original_domain.0 == root.authority().epoch
            && root.hard_deadline().accepts(original_domain, now).is_ok()
            && dispatch
                .original_deadline()
                .accepts(original_domain, now)
                .is_ok()
            && !guard.is_ticket_canceled(),
        "exact witness canceled or original deadline elapsed during projection",
    )?;
    Ok(observed)
}
