use super::*;
use rz_contracts::pals_adaptive::RoleTicketEnvelope;
use rz_eval::pals_model::virtual_review::{
    PalsVirtualReviewInput, ReviewQueryPurpose, ReviewRecordKind,
};
use rz_eval::pals_model::{PalsCandidateToken, PalsRole};
use rz_search::cpu::CpuScoreScope;
use rz_search::pals::engine::RecordKind;
use rz_search::pals::review_context::ReviewView;

/// 원 Rules repetition identity의 기존 generation grouping domain이다.
/// 모델/평가 캐시 또는 Rules semantic state digest와 혼용하지 않는다.
pub(crate) fn generation_transposition_sha256(position: &Position) -> String {
    use sha2::{Digest, Sha256};
    use std::hash::{Hash, Hasher};

    struct IdentityHash(Sha256);
    impl Hasher for IdentityHash {
        fn finish(&self) -> u64 {
            u64::from_le_bytes(self.0.clone().finalize()[..8].try_into().unwrap())
        }
        fn write(&mut self, b: &[u8]) {
            self.0.update(b);
        }
    }
    let mut transposition = IdentityHash(Sha256::new());
    transposition.write(b"rz-pals-generation-transposition/1\0");
    position
        .snapshot()
        .repetition_identity()
        .hash(&mut transposition);
    format!("{:x}", transposition.0.finalize())
}

#[derive(Clone, Debug)]
enum CaptureWorkScope {
    Native(rz_contracts::RequestId),
    Own(rz_contracts::pals_adaptive::SliceGrant),
    External(rz_search::external_cpu::worker::ExternalWorkerScope),
}
/// Immutable source of audit IDs. Caller text is not a worker/job/slice proof.
#[derive(Clone, Debug)]
pub struct GenerationCaptureWorkScopeV3 {
    source: CaptureWorkScope,
}
impl GenerationCaptureWorkScopeV3 {
    pub fn from_native_ticket(ticket: &RoleTicketEnvelope) -> Self {
        Self {
            source: CaptureWorkScope::Native(ticket.id),
        }
    }
    pub fn from_cpu_report(
        report: &rz_search::cpu_worker::CpuSliceReport,
    ) -> Result<Self, ArenaError> {
        check(
            !report.admission_rejected && report.actual_work.is_some() && report.result.is_ok(),
            "unobserved/rejected CPU packet cannot issue capture scope",
        )?;
        Ok(Self {
            source: CaptureWorkScope::Own(report.grant),
        })
    }
    pub fn from_external_report(
        report: &rz_search::external_cpu::worker::ExternalWorkerReport,
    ) -> Result<Self, ArenaError> {
        check(
            report.result.is_ok()
                && report.go_issue == rz_search::external_cpu::ExternalGoIssue::Issued,
            "unissued/failed external packet cannot issue capture scope",
        )?;
        Ok(Self {
            source: CaptureWorkScope::External(report.scope),
        })
    }
    pub(super) fn identifiers(
        &self,
        ticket: &RoleTicketEnvelope,
    ) -> Result<(String, String, String), ArenaError> {
        match self.source {
            CaptureWorkScope::Native(id) => {
                check(
                    id == ticket.id,
                    "capture native scope belongs to another ticket",
                )?;
                Ok((
                    format!("native-owner-{}", id.epoch.0),
                    format!("question-{}-{}", id.epoch.0, id.sequence),
                    format!("request-{}-{}", id.epoch.0, id.sequence),
                ))
            }
            CaptureWorkScope::Own(grant) => {
                check(
                    grant.job.authority == ticket.authority,
                    "capture CPU packet belongs to another root",
                )?;
                Ok((
                    format!("cpu-owner-{}-{}", ticket.authority.epoch.0, grant.owner.0),
                    format!(
                        "cpu-job-{}-{}",
                        grant.job.authority.epoch.0, grant.job.sequence
                    ),
                    format!(
                        "cpu-slice-{}-{}",
                        grant.execution.epoch.0, grant.execution.sequence
                    ),
                ))
            }
            CaptureWorkScope::External(scope) => {
                check(
                    scope.job.authority == ticket.authority,
                    "capture external packet belongs to another root",
                )?;
                Ok((
                    format!(
                        "external-owner-{}-{}",
                        ticket.authority.epoch.0, scope.owner.0
                    ),
                    format!(
                        "external-job-{}-{}",
                        scope.job.authority.epoch.0, scope.job.sequence
                    ),
                    format!(
                        "external-slice-{}-{}",
                        scope.execution.epoch.0, scope.execution.sequence
                    ),
                ))
            }
        }
    }
}

/// The Collector cannot declare support for a V action with caller booleans.
/// Search freezes capabilities with the actual Task question before dispatch.
#[derive(Clone, Debug)]
pub struct GenerationActionSupportV3 {
    validator: Option<(
        rz_contracts::pals::SearchAuthority,
        rz_contracts::Digest,
        [u8; 32],
        [bool; 7],
    )>,
}
impl GenerationActionSupportV3 {
    pub fn non_validator() -> Self {
        Self { validator: None }
    }
    pub fn from_validator_question(
        question:&rz_search::pals::engine::adaptive_port::validator_port::AdaptiveValidatorQuestion,
    ) -> Self {
        Self {
            validator: Some((
                question.parent().authority,
                question.input().question_digest,
                question.input().view.seal_digest(),
                question.support_mask(),
            )),
        }
    }
    fn for_ticket(&self, ticket: &RoleTicketEnvelope) -> Result<[bool; 7], ArenaError> {
        match (ticket.role, &self.validator) {
            (rz_contracts::pals::Role::Validator, Some((authority, question, view, mask))) => {
                check(
                    *authority == ticket.authority
                        && *question == ticket.question
                        && *view == ticket.view.view_digest.0,
                    "V capture support belongs to another actual question/view",
                )?;
                Ok(*mask)
            }
            (rz_contracts::pals::Role::Proposer | rz_contracts::pals::Role::Critic, None) => {
                Ok([false; 7])
            }
            _ => Err(invalid(
                "actual V support capability required only for Validator",
            )),
        }
    }
}

/// Preserve the original ticket namespace in every actual-source receipt.
pub(super) fn generation_role_ticket_json_v3(ticket: &RoleTicketEnvelope) -> serde_json::Value {
    serde_json::json!({"epoch":ticket.id.epoch.0,"request_sequence":ticket.id.sequence,"game":ticket.authority.game.0,"root":ticket.authority.root.0,"question_sha256":hex(&ticket.question.0),
        "source_view_sha256":hex(&ticket.view.view_digest.0),"view_sequence":ticket.view.view_sequence,"context_policy_sha256":hex(&ticket.view.context_policy.0),
        "representation_input_sha256":hex(&ticket.representation.input.0),"representation_history_sha256":hex(&ticket.representation.history.0),"representation_model_sha256":hex(&ticket.representation.model.0),"representation_encoding_sha256":hex(&ticket.representation.encoding.0),
        "frozen_epoch":ticket.representation.frozen_epoch,"record_revision":ticket.representation.record_revision,"deadline_tick":ticket.original_deadline.at.0})
}

/// Context is frozen by the manager for the original submitted question. IDs
/// denote actual job/slice ownership or that exact neural ticket's own work.
#[derive(Clone, Debug)]
pub struct GenerationCaptureContextV3 {
    pub game_id: String,
    pub opening_id: String,
    pub line_genealogy_id: String,
    pub model_epoch: u64,
    pub context_epoch: u64,
    pub capture_sequence: u64,
    pub work_scope: GenerationCaptureWorkScopeV3,
    pub actual_position: Position,
    pub actual_history: Vec<BoardMove>,
    pub action_support: GenerationActionSupportV3,
}
pub fn virtual_review_json_v3(
    input: &PalsVirtualReviewInput,
) -> Result<serde_json::Value, ArenaError> {
    input.validate().map_err(|e| invalid(e.to_string()))?;
    let components = |m: &PalsCandidateToken| [m.from, m.to, m.promotion];
    Ok(serde_json::json!({"schema":input.schema,"role":input.role,
        "anchors":input.anchors,"records":input.records.iter().map(|r|serde_json::json!({
            "kind":r.kind,"perspective":r.perspective,"completed_iteration":r.completed_iteration,"rules_terminal":r.rules_terminal,
            "line":r.line.iter().map(components).collect::<Vec<_>>(),"anchor":r.anchor,"parent":r.parent,"supersedes":r.supersedes})).collect::<Vec<_>>(),
        "query":{"purpose":input.query.purpose,"perspective":input.query.perspective,"anchors":input.query.anchors,"counter_present":input.query.counter_present,
            "prefix":input.query.prefix.iter().map(components).collect::<Vec<_>>(),"proposal":input.query.proposal.iter().map(components).collect::<Vec<_>>(),"counter":input.query.counter.iter().map(components).collect::<Vec<_>>()},
        "candidates":input.candidates.iter().map(components).collect::<Vec<_>>(),"divergences":input.divergences,"token_budget":input.token_budget}))
}
/// This function captures input only. It does not issue actual producer
/// authority: the native observer must later join physical completion and ack.
pub fn capture_submitted_review_v3(
    ticket: &RoleTicketEnvelope,
    view: &ReviewView,
    input: &PalsVirtualReviewInput,
    context: &GenerationCaptureContextV3,
) -> Result<GenerationInputV3, ArenaError> {
    let mut captured = capture_submitted_review_common_fields(ticket, view, input, context)?;
    captured.divergence_record_ids = view
        .query()
        .divergences
        .iter()
        .map(|slot| {
            captured
                .selected_view
                .records
                .get(*slot)
                .filter(|r| r.kind == GenerationRecordKindV3::Counter)
                .map(|r| r.record_id.clone())
                .ok_or_else(|| invalid("submitted divergence index absent/noncounter"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    captured.validate()?;
    Ok(captured)
}

/// 공통 실제 Rules/query/tensor capture 계산이다. 이 중간 projection은
/// legacy V3나 새 Rovent 입력으로 발행하지 않는다. 각 명시 wrapper가 자기
/// divergence 의미를 추가하고 전체 형식 검증을 완료해야 한다.
pub(crate) fn capture_submitted_review_common_fields(
    ticket: &RoleTicketEnvelope,
    view: &ReviewView,
    input: &PalsVirtualReviewInput,
    context: &GenerationCaptureContextV3,
) -> Result<GenerationInputV3, ArenaError> {
    check(
        ticket.view.view_digest.0 == view.seal_digest()
            && ticket.view.view_sequence == view.sequence()
            && ticket.view.context_policy.0 == view.policy_digest()
            && ticket.view.actual_tokens == view.token_count()
            && ticket.view.length_bucket == view.length_bucket()
            && ticket.view.selected_records as usize == view.records().len()
            && ticket.view.anchors as usize == view.anchors().len()
            && view.records().len() == view.records_metadata().len()
            && input.records.len() == view.records().len()
            && input.anchors.len() == view.anchors().len(),
        "original ticket/view/allocation capture mismatch",
    )?;
    let role = match input.role {
        PalsRole::Proposer => GenerationRoleV3::Proposer,
        PalsRole::Critic => GenerationRoleV3::Critic,
        PalsRole::Validator => GenerationRoleV3::Validator,
    };
    check(
        ticket.role
            == match role {
                GenerationRoleV3::Proposer => rz_contracts::pals::Role::Proposer,
                GenerationRoleV3::Critic => rz_contracts::pals::Role::Critic,
                GenerationRoleV3::Validator => rz_contracts::pals::Role::Validator,
            },
        "captured role differs from original ticket",
    )?;
    let mut anchors = vec![];
    for anchor in view.anchors() {
        let position =
            Position::from_snapshot(&anchor.snapshot, rz_position::PositionLimits::default())
                .map_err(|e| invalid(e.to_string()))?;
        anchors.push(exact_rules_anchor(&position)?);
    }
    let root = exact_rules_anchor(&context.actual_position)?;
    check(
        anchors.first() == Some(&root),
        "submitted review root is not the actual game root",
    )?;
    check(
        context
            .actual_position
            .snapshot()
            .uci_replay(16_384)
            .map_err(|e| invalid(e.to_string()))?
            .moves
            == context.actual_history,
        "capture actual history differs from exact played root",
    )?;
    let mut query_position = Position::from_snapshot(
        &view
            .anchors()
            .get(input.query.anchors[0] as usize)
            .ok_or_else(|| invalid("query anchor outside selected view"))?
            .snapshot,
        rz_position::PositionLimits::default(),
    )
    .map_err(|e| invalid(e.to_string()))?;
    for movement in &view.query().prefix {
        query_position
            .make_move(*movement)
            .map_err(|e| invalid(e.to_string()))?;
    }
    let query_anchor = exact_rules_anchor(&query_position)?;
    let expected_legal = pack(&query_position.legal_moves())?;
    let prepared_legal = input
        .candidates
        .iter()
        .map(|m| u16::from(m.from) | (u16::from(m.to) << 6) | (u16::from(m.promotion) << 12))
        .collect::<Vec<_>>();
    check(
        (input.query.purpose == ReviewQueryPurpose::TaskPolicy
            || input.query.purpose == ReviewQueryPurpose::DivergencePolicy)
            && prepared_legal.is_empty()
            || expected_legal.len() == prepared_legal.len()
                && expected_legal.iter().all(|m| prepared_legal.contains(m)),
        "captured candidates are not complete legal query endpoint moves",
    )?;
    let mut records = Vec::with_capacity(view.records().len());
    for ((record, metadata), encoded) in view
        .records()
        .iter()
        .zip(view.records_metadata())
        .zip(&input.records)
    {
        let mut line_endpoint = Position::from_snapshot(
            &view
                .anchors()
                .get(metadata.anchor as usize)
                .ok_or_else(|| invalid("record anchor outside selected view"))?
                .snapshot,
            rz_position::PositionLimits::default(),
        )
        .map_err(|e| invalid(e.to_string()))?;
        for movement in &record.line {
            line_endpoint
                .make_move(*movement)
                .map_err(|e| invalid(e.to_string()))?;
        }
        let rules_terminal = matches!(
            line_endpoint
                .classify_position()
                .map_err(|e| invalid(e.to_string()))?
                .play_status,
            rz_position::PlayStatus::Terminal { .. }
        );
        let kind = match record.kind {
            RecordKind::Proposal => GenerationRecordKindV3::Proposal,
            RecordKind::Counterexample => GenerationRecordKindV3::Counter,
            RecordKind::Repair => GenerationRecordKindV3::Repair,
            RecordKind::CpuVerification => GenerationRecordKindV3::Continuation,
        };
        let encoded_kind = match kind {
            GenerationRecordKindV3::Proposal => ReviewRecordKind::Proposal,
            GenerationRecordKindV3::Counter => ReviewRecordKind::Counterexample,
            GenerationRecordKindV3::Repair => ReviewRecordKind::Repair,
            GenerationRecordKindV3::Continuation => ReviewRecordKind::CpuVerification,
        };
        check(
            encoded.kind == encoded_kind
                && encoded.anchor == metadata.anchor
                && encoded.parent == metadata.parent
                && encoded.supersedes == metadata.supersedes
                && encoded.completed_iteration
                    == (record.score_scope == Some(CpuScoreScope::CompletedIteration))
                && encoded.rules_terminal == rules_terminal,
            "native prepared record differs from selected record",
        )?;
        records.push(GenerationSelectedRecordV3 {
            record_id: format!("record-{}-{}", view.generation(), metadata.source_revision),
            kind,
            anchor: metadata.anchor,
            moves: pack(&record.line)?,
            parent: metadata.parent,
            supersedes: metadata.supersedes,
            // The view already contains public evidence when this exact
            // submission is captured. Its Store revision is preserved by the
            // record ID/source view; it is not a collector event counter.
            observed_sequence: context.capture_sequence,
            perspective: if record.perspective == Color::White {
                1
            } else {
                -1
            },
            completed_iteration: encoded.completed_iteration,
            rules_terminal: encoded.rules_terminal,
        });
    }
    let purpose = match input.query.purpose {
        ReviewQueryPurpose::ProposePolicy => "propose_policy",
        ReviewQueryPurpose::ReplyPolicy => "reply_policy",
        ReviewQueryPurpose::RepairPolicy => "repair_policy",
        ReviewQueryPurpose::DivergencePolicy => "divergence_policy",
        ReviewQueryPurpose::ValueFresh => "value_fresh",
        ReviewQueryPurpose::TaskPolicy => "task_policy",
    };
    let component_bits = |m: &PalsCandidateToken| {
        u16::from(m.from) | (u16::from(m.to) << 6) | (u16::from(m.promotion) << 12)
    };
    check(
        input.query.counter_present == view.query().counterexample.is_some(),
        "native query lost explicit counter presence",
    )?;
    let query = GenerationSelectedQueryV3 {
        purpose: purpose.into(),
        anchors: input.query.anchors,
        prefix: input.query.prefix.iter().map(component_bits).collect(),
        proposal: input.query.proposal.iter().map(component_bits).collect(),
        counter: input.query.counter.iter().map(component_bits).collect(),
        counter_present: input.query.counter_present,
        white_to_move: input.query.perspective == 1,
    };
    let selected_view = GenerationSelectedViewV3 {
        context_policy_sha256: hex(&view.policy_digest()),
        anchors,
        records,
        query,
        token_budget: input.token_budget,
        length_bucket: view.length_bucket(),
    };
    let (worker_id, job_id, slice_id) = context.work_scope.identifiers(ticket)?;
    let captured = GenerationInputV3 {
        game_id: context.game_id.clone(),
        opening_id: context.opening_id.clone(),
        line_genealogy_id: context.line_genealogy_id.clone(),
        transposition_sha256: generation_transposition_sha256(&query_position),
        role,
        model_epoch: context.model_epoch,
        context_epoch: context.context_epoch,
        capture_sequence: context.capture_sequence,
        question_id: hex(&ticket.question.0),
        worker_id,
        job_id,
        slice_id,
        rules_anchor: query_anchor,
        actual_history: pack(&context.actual_history)?,
        view_sha256: selected_view.sha256()?,
        selected_view,
        legal_moves: input.candidates.iter().map(component_bits).collect(),
        action_support_mask: context.action_support.for_ticket(ticket)?,
        encoded_model_input: virtual_review_json_v3(input)?,
        divergence_record_ids: Vec::new(),
    };
    Ok(captured)
}

#[cfg(feature = "pals-collection-onnx")]
mod observer;
#[cfg(feature = "pals-collection-onnx")]
pub use observer::*;

/// Teacher completion is consumed from the private external owner packet.
/// Bind the original job before dispatch and join that binding here; reported
/// nodes, CP/mate and speculative partial tuples do not become value labels.
pub struct TeacherInputBindingV3 {
    input_sha256: String,
    job: rz_contracts::pals_adaptive::ExternalCpuJobEnvelope,
    grant: rz_contracts::pals_adaptive::ExternalSliceGrant,
    scope: rz_search::external_cpu::worker::ExternalWorkerScope,
    submitted: rz_search::external_cpu::worker::ExternalTeacherDispatch,
}
#[derive(Default)]
pub struct TeacherBindingRegistryV3 {
    executions: BTreeSet<(u64, u64, String)>,
}
impl TeacherBindingRegistryV3 {
    pub(crate) fn bind_actual_submission(
        &mut self,
        record: &GenerationRecordV3,
        job: &rz_search::external_cpu::worker::ExternalJob,
        submitted: &rz_search::external_cpu::worker::ExternalTeacherDispatch,
    ) -> Result<TeacherInputBindingV3, ArenaError> {
        record.validate()?;
        check(
            record.input.role != GenerationRoleV3::Validator
                && exact_rules_anchor(&job.position)? == record.input.rules_anchor
                && submitted.matches_job(job)
                && self.executions.len() < 128
                && !self.executions.contains(&(
                    job.execution.epoch.0,
                    job.execution.sequence,
                    record.input_sha256.clone(),
                )),
            "teacher binding requires actual owner submission, coherent input/job and unique input/execution",
        )?;
        self.executions.insert((
            job.execution.epoch.0,
            job.execution.sequence,
            record.input_sha256.clone(),
        ));
        Ok(TeacherInputBindingV3 {
            input_sha256: record.input_sha256.clone(),
            job: job.envelope,
            grant: job.grant,
            scope: job.scope(),
            submitted: submitted.clone(),
        })
    }
}
#[allow(
    clippy::too_many_arguments,
    reason = "The original teacher job, completion, chronology and separately pinned binary and profile authorities are validated independently"
)]
pub fn attach_completed_teacher_policy_v3(
    record: &mut GenerationRecordV3,
    binding: &TeacherInputBindingV3,
    original: &rz_search::external_cpu::worker::ExternalJob,
    completion: &rz_search::external_cpu::worker::ExternalWorkerCompletion,
    observed_sequence: u64,
    teacher_binary_sha256: &str,
    teacher_profile_sha256: &str,
    evidence: &mut BTreeMap<String, serde_json::Value>,
) -> Result<bool, ArenaError> {
    use rz_search::cpu_checker::ExternalCompletion;
    use rz_search::external_cpu::ExternalGoIssue;
    let actual = completion.report();
    record.validate()?;
    check(
        record.input.role != GenerationRoleV3::Validator,
        "V cannot receive teacher policy",
    )?;
    check(
        binding.input_sha256 == record.input_sha256
            && binding.job == actual.job
            && binding.grant == actual.grant
            && binding.scope == actual.scope
            && actual.job == original.envelope
            && actual.grant == original.grant
            && actual.scope == original.scope()
            && original.envelope.question.0.iter().any(|b| *b != 0)
            && exact_rules_anchor(&original.position)? == record.input.rules_anchor,
        "teacher report job/grant/owner/anchor mismatch",
    )?;
    if !completion.matches_dispatch(&binding.submitted) {
        return Ok(false);
    }
    let Ok(report) = &actual.result else {
        return Ok(false);
    };
    if report.go_issue != ExternalGoIssue::Issued
        || actual.go_issue != ExternalGoIssue::Issued
        || !report.policy_target_known
        || !report.readiness_barrier_completed
        || report.report.completion != ExternalCompletion::BestMove
    {
        return Ok(false);
    }
    check(
        report.report.identity.binary_sha256 == teacher_binary_sha256
            && generation_sha(teacher_profile_sha256),
        "teacher pin differs from collection registration",
    )?;
    let mut opinions = report.opinions.clone();
    opinions.sort_by_key(|o| o.rank);
    check(
        !opinions.is_empty()
            && opinions.len() <= 4
            && opinions.iter().enumerate().all(|(i, o)| {
                o.rank as usize == i + 1 && o.depth > 0 && o.depth == opinions[0].depth
            }),
        "teacher rank/depth set is incomplete",
    )?;
    for opinion in &opinions {
        let mut replay = original.position.clone();
        for m in &opinion.pv {
            replay.make_move(*m).map_err(|e| invalid(e.to_string()))?;
        }
    }
    let ranked_moves = pack(
        &opinions
            .iter()
            .filter_map(|o| o.pv.first().copied())
            .collect::<Vec<_>>(),
    )?;
    let mut target = GenerationPolicyTargetV3 {
        observed_sequence,
        question_id: record.input.question_id.clone(),
        view_sha256: record.input.view_sha256.clone(),
        raw_evidence_sha256: "0".repeat(64),
        source: "stockfish19_completed_multipv".into(),
        completed: true,
        stale: false,
        anchor: record.input.rules_anchor.clone(),
        ranked_moves,
        teacher_binary_sha256: teacher_binary_sha256.into(),
        teacher_profile_sha256: teacher_profile_sha256.into(),
        go_sequence: report.report.request_id,
        completed_depth: u32::from(opinions[0].depth),
    };
    target.probabilities(&record.input)?;
    check(
        record.future_targets.policy.is_none(),
        "duplicate conditional teacher policy",
    )?;
    target.raw_evidence_sha256 =
        seal_head_evidence(&record.input_sha256, "policy", &target, evidence)?;
    record.future_targets.policy = Some(target);
    record.validate()?;
    Ok(true)
}
