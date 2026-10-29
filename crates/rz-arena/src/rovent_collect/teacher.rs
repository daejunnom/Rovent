use super::*;
use rz_search::cpu_checker::{ExternalProcessIdentity, ExternalRawScore};
use rz_search::external_cpu::{
    ExternalEngineProfile, ExternalEnginePurpose, ExternalGoIssue, ProfiledExternalCompletion,
    ProfiledExternalDispatch, ProfiledExternalJob, ProfiledExternalScope,
};
use rz_uci::pals_checker_profile::rovent::RoventExternalProfile;
use rz_uci::rovent_external_launch::{PreparedRoventExternalLaunch, RoventExternalOwner};
mod collection;
pub use collection::RoventCollectionTeacherBinding;
mod session;
mod v2_actor_preflight;
#[cfg(feature = "pals-collection-onnx")]
mod v2_session;
pub use session::*;
mod pump;
pub use pump::*;

fn process_json(p: ExternalProcessIdentity) -> Value {
    json!({"pid":p.pid,"process_group":p.process_group,"proc_start_ticks":p.proc_start_ticks})
}
fn score_json(s: ExternalRawScore) -> Value {
    match s {
        ExternalRawScore::Centipawns(v) => json!({"kind":"centipawns","value":v}),
        ExternalRawScore::MateMoves(v) => json!({"kind":"mate_moves","value":v}),
        ExternalRawScore::Unknown => json!({"kind":"unknown"}),
    }
}
fn question_json(q: rz_contracts::rovent::QuestionEnvelope) -> Value {
    json!({"process_epoch":q.id.authority.epoch.0,"game_generation":q.id.authority.game.0,"root_generation":q.id.authority.root.0,
        "implementation_sha256":hex(q.id.authority.implementation),"sequence":q.id.sequence,
        "parent":q.parent.map(|p|json!({"process_epoch":p.authority.epoch.0,"game_generation":p.authority.game.0,"root_generation":p.authority.root.0,"sequence":p.sequence})),
        "sender":format!("{:?}",q.sender),"recipient":format!("{:?}",q.recipient),"purpose":format!("{:?}",q.purpose),
        "anchor":{"owner":q.anchor.owner.0,"revision":q.anchor.revision.0,"semantic":hex(q.anchor.semantic)},"anchor_root_ply":q.anchor_root_ply.0,
        "original_first_move":q.original_first_move.map(|m|m.bits()),"prefix_digest":hex(q.prefix_digest),
        "line_digest":q.line_digest.map(hex),"line_offset":q.line_offset.0,"question_digest":hex(q.question_digest),
        "evidence_revision":q.evidence_revision,"root_budget":hex(q.root_budget),"deadline_clock":q.original_deadline.clock.0.0,"deadline_tick":q.original_deadline.at.0,
        "view":{"view_digest":hex(q.view.view_digest),"view_sequence":q.view.view_sequence,"context_policy":hex(q.view.context_policy),
            "actual_tokens":q.view.actual_tokens,"selected_records":q.view.selected_records,"anchors":q.view.anchors,"length_bucket":q.view.length_bucket}})
}
/// A declaration joined to the existing retained launch. It is deliberately
/// not a running teacher, go receipt, completion or a training authority.
pub struct RoventRegisteredTeacher {
    actor_id: String,
    pin: RoventTeacherPin,
}
impl RoventRegisteredTeacher {
    pub fn from_prepared(
        manifest: &RoventCorpusManifest,
        actor_id: &str,
        launch: &PreparedRoventExternalLaunch,
        original_profile_bytes: &[u8],
    ) -> Result<Self, ArenaError> {
        let pin = manifest
            .teachers
            .get(actor_id)
            .ok_or_else(|| invalid("teacher actor absent from corpus manifest"))?;
        let loaded =
            RoventExternalProfile::from_verified_bytes(original_profile_bytes, &pin.profile_sha256)
                .map_err(|e| invalid(e.to_string()))?;
        let p = loaded.profile();
        let family = match p.engine {
            ExternalEngineProfile::Stockfish19EmbeddedNnue => "stockfish",
            ExternalEngineProfile::Reckless090EmbeddedNnue => "reckless",
        };
        let assets = p
            .identity
            .assets
            .iter()
            .map(|a| (a.purpose.clone(), a.sha256.clone()))
            .collect::<BTreeMap<_, _>>();
        check(
            p == launch.profile()
                && p.purpose == ExternalEnginePurpose::Teacher
                && pin.engine == family
                && p.engine.source_commit() == pin.source_commit
                && p.identity.binary_sha256 == pin.binary_sha256
                && launch.binary().sha256 == pin.binary_sha256
                && launch.binary().bytes == pin.binary_bytes
                && loaded.registration().canonical_sha256 == pin.canonical_sha256
                && launch.canonical_sha256() == pin.canonical_sha256
                && launch.capability_sha256() == pin.capability_sha256
                && p.identity.options == pin.options
                && assets.len() == p.identity.assets.len()
                && assets == pin.assets,
            "teacher recipe/profile/source/binary/options/assets do not match retained launch",
        )?;
        Ok(Self {
            actor_id: actor_id.into(),
            pin: pin.clone(),
        })
    }
}
/// Created before the teacher go is submitted, against the frozen student
/// input and exact Rules position. The returned owner dispatch must later
/// match this original job; the worker's private completion remains mandatory.
pub struct RoventTeacherBinding {
    input_sha256: String,
    anchor: GenerationRulesAnchorV3,
    actor_id: String,
    pin: RoventTeacherPin,
    scope: ProfiledExternalScope,
    question: rz_contracts::rovent::QuestionEnvelope,
    constraint: rz_search::external_cpu::ExternalGoConstraint,
    root_restriction: Digest,
    process: ExternalProcessIdentity,
    root_admission: std::sync::Arc<rz_uci::rovent_root_admission::RoventRootAdmission>,
    raw: Value,
}
impl RoventCorpusProducer {
    pub fn bind_teacher(
        &self,
        row: &RoventRetiredRecord,
        registered: &RoventRegisteredTeacher,
        owner: &RoventExternalOwner,
        job: &ProfiledExternalJob,
    ) -> Result<RoventTeacherBinding, ArenaError> {
        self.bind_teacher_input(&row.submitted_input_seal(), registered, owner, job)
    }
    /// NN이 진행 중이어도 원 제출 입력에 교사 go를 bind할 수 있다.
    /// 이 값만으로 policy target이나 완료 record를 발급하지 않는다.
    pub fn bind_teacher_input(
        &self,
        input: &RoventSubmittedInputSeal,
        registered: &RoventRegisteredTeacher,
        owner: &RoventExternalOwner,
        job: &ProfiledExternalJob,
    ) -> Result<RoventTeacherBinding, ArenaError> {
        let startup = owner
            .verified_startup()
            .map_err(|e| invalid(e.to_string()))?
            .ok_or_else(|| invalid("teacher actual startup missing"))?;
        let root_admission = owner
            .actual_root_admission()
            .ok_or_else(|| invalid("teacher requires the original actual-go root admission"))?;
        let anchor = exact_rules_anchor(&job.position)?;
        let root_restriction = rz_search::external_cpu::worker::external_root_restriction_digest(
            job.root_moves.as_deref(),
        );
        check(
            input.manifest_sha256 == self.manifest_sha256
                && input.teacher_actor_id.as_ref() == Some(&registered.actor_id)
                && input.role != GenerationRoleV3::Validator
                && self.manifest.teachers.get(&registered.actor_id) == Some(&registered.pin)
                && anchor == input.anchor
                && job.question.id.authority == input.ticket.authority
                && root_admission.authority() == input.ticket.authority
                && hex(root_admission.startup_lock_digest()) == startup.input_lock_sha256()
                && root_admission.hard_deadline() == input.ticket.original_deadline
                && job.question.root_budget == root_admission.root_budget_digest()
                && job.question.original_deadline.clock == root_admission.hard_deadline().clock
                && job.question.original_deadline.at <= root_admission.hard_deadline().at
                && job.question.anchor.semantic
                    == rz_position::contracts::rules_snapshot_semantic_digest(
                        &job.position.snapshot(),
                    )
                && startup.canonical_sha256() == registered.pin.canonical_sha256
                && startup.capability_sha256() == registered.pin.capability_sha256,
            "teacher job does not bind exact submitted student input/root/actor",
        )?;
        let scope = job.scope();
        let process = startup.process_identity();
        let raw = json!({"question_sequence":job.question.id.sequence,"question_digest":hex(job.question.question_digest),
            "root_budget":hex(job.question.root_budget),"root_restriction":hex(root_restriction),
            "execution_epoch":job.execution.epoch.0,"execution_sequence":job.execution.sequence,
            "owner":job.owner.0,"process":process_json(process),"requested":job.constraint,"question":question_json(job.question),
            "position_anchor":anchor,"root_moves":job.root_moves.as_ref().map(|m|packed(m)).transpose()?,"mode":format!("{:?}",job.mode),
            "soft_quantum_ns":job.soft_quantum.as_nanos().min(u128::from(u64::MAX)) as u64,"context_token_budget":job.context_token_budget,
            "root_descriptor":root_admission.descriptor(),"root_descriptor_sha256":hex(root_admission.digest())});
        Ok(RoventTeacherBinding {
            input_sha256: input.input_sha256.clone(),
            anchor,
            actor_id: registered.actor_id.clone(),
            pin: registered.pin.clone(),
            scope,
            question: job.question,
            constraint: job.constraint,
            root_restriction,
            process,
            root_admission: std::sync::Arc::clone(root_admission),
            raw,
        })
    }
    pub fn attach_teacher_policy(
        &mut self,
        row: &mut RoventRetiredRecord,
        binding: RoventTeacherBinding,
        owner: &RoventExternalOwner,
        dispatch: &ProfiledExternalDispatch,
        completion: &ProfiledExternalCompletion,
    ) -> Result<(), ArenaError> {
        owner
            .validate_actual_completion(completion, dispatch)
            .map_err(|e| invalid(e.to_string()))?;
        check(
            row.record.targets.policy.is_none()
                && binding.input_sha256 == row.record.input_sha256
                && binding.anchor == row.record.input.rules_anchor
                && std::sync::Arc::ptr_eq(&binding.root_admission, row.admission.root_admission())
                && dispatch.scope() == binding.scope
                && dispatch.question() == binding.question
                && dispatch.constraint() == binding.constraint
                && dispatch.root_restriction() == binding.root_restriction
                && dispatch.process_identity() == binding.process
                && completion.go_issue() == ExternalGoIssue::Issued,
            "teacher original input/job/dispatch mismatch",
        )?;
        let outcome = completion
            .result()
            .as_ref()
            .map_err(|e| invalid(e.to_string()))?;
        let startup = owner
            .verified_startup()
            .map_err(|e| invalid(e.to_string()))?
            .ok_or_else(|| invalid("teacher startup lost"))?;
        let attempt = completion
            .attempt()
            .ok_or_else(|| invalid("teacher missing original physical attempt"))?;
        self.apply_completed_teacher_policy(
            row,
            binding.actor_id,
            binding.pin,
            binding.raw,
            &startup,
            outcome,
            attempt,
            completion.timing(),
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn apply_completed_teacher_policy(
        &mut self,
        row: &mut RoventRetiredRecord,
        actor_id: String,
        pin: RoventTeacherPin,
        binding_raw: Value,
        startup: &rz_uci::rovent_external_launch::VerifiedRoventExternalStartup,
        outcome: &rz_search::external_cpu::ProfiledExternalOutcome,
        attempt: &rz_search::cpu_checker::CheckerAttempt,
        timing: rz_search::external_cpu::ProfiledExternalTiming,
    ) -> Result<(), ArenaError> {
        let r = &outcome.report;
        let raw = &r.raw;
        check(
            r.purpose == ExternalEnginePurpose::Teacher
                && r.profile.source_commit() == pin.source_commit
                && raw.identity.binary_sha256 == pin.binary_sha256
                && raw.identity.options == pin.options
                && outcome.conditional_policy_target_known
                && r.readiness_barrier_completed
                && !outcome.opinions.is_empty()
                && outcome.opinions.len() <= 4,
            "failed/partial/stale teacher report has no conditional policy target",
        )?;
        let mut opinions = outcome.opinions.iter().collect::<Vec<_>>();
        opinions.sort_by_key(|o| o.rank);
        let mut ranked = Vec::new();
        let mut depth = u16::MAX;
        for (slot, o) in opinions.iter().enumerate() {
            check(
                o.rank as usize == slot + 1 && o.depth > 0 && !o.pv.is_empty(),
                "incomplete teacher rank/depth/PV",
            )?;
            let first = Move16::pack(o.pv[0])
                .map_err(|e| invalid(e.to_string()))?
                .bits();
            check(
                row.record.input.legal_moves.contains(&first) && !ranked.contains(&first),
                "teacher first move is illegal/duplicated",
            )?;
            // The original worker validates every PV suffix from its retained
            // exact job position. owner.validate_completion above joins that
            // private worker completion and its unchanged raw report.
            ranked.push(first);
            depth = depth.min(o.depth);
        }
        let sequence = self.next()?;
        let raw_report = json!({"identity":raw.identity,"observed_uci":{"name":raw.observed_uci.name,"author":raw.observed_uci.author},
            "request_id":raw.request_id,"best_move":raw.best_move.map(|m|Move16::pack(m).map(|v|v.bits())).transpose().map_err(|e|invalid(e.to_string()))?,
            "pv":packed(&raw.pv)?,"score":score_json(raw.score),"bound":format!("{:?}",raw.bound),"wdl_per_mille":raw.wdl_per_mille,
            "perspective":format!("{:?}",raw.perspective),"requested_depth":raw.requested_depth,"reported_depth":raw.reported_depth,
            "seldepth":raw.seldepth,"root_restricted":raw.root_restricted,"completion":format!("{:?}",raw.completion),
            "work":{"nodes":raw.work.nodes,"qnodes":raw.work.qnodes,"tt_hits":raw.work.tt_hits},
            "elapsed_ns":raw.elapsed.as_nanos().min(u128::from(u64::MAX)) as u64});
        let opinions_json=opinions.iter().map(|o|Ok(json!({"rank":o.rank,"depth":o.depth,"seldepth":o.seldepth,
            "pv":packed(&o.pv)?,"score":score_json(o.score),"bound":format!("{:?}",o.bound),"reported_wdl":o.reported_wdl}))).collect::<Result<Vec<_>,ArenaError>>()?;
        let physical = attempt
            .external
            .as_ref()
            .ok_or_else(|| invalid("teacher physical process receipt missing"))?;
        let p = physical.process;
        let attempt_json = json!({"elapsed_ns":attempt.elapsed.as_nanos().min(u128::from(u64::MAX)) as u64,
            "work":{"nodes":attempt.work.nodes,"qnodes":attempt.work.qnodes,"tt_hits":attempt.work.tt_hits},
            "request_id":physical.request_id,"partial_report_matches":physical.partial_report.as_ref()==Some(raw),
            "process":{"identity":p.process_identity.map(process_json),"stop_sent":p.stop_sent,"quit_sent":p.quit_sent,
                "exit_observed":p.exit_observed,"stdout_drained":p.stdout_drained,"stderr_drained":p.stderr_drained,
                "exit_code":p.exit_code,"exit_signal":p.exit_signal,"cleanup_complete":p.cleanup_complete,
                "quarantined":p.quarantined,"ownership_lost":p.ownership_lost,"stdout_bytes":p.stdout_bytes,"stderr_bytes":p.stderr_bytes}});
        let body = json!({"schema":HEAD_DOMAIN,"kind":"teacher_policy","actor_id":actor_id,
            "input_sha256":row.record.input_sha256,"anchor":row.record.input.rules_anchor,"teacher_pin":pin,"binding":binding_raw,
            "completion":{"profile":r.profile,"purpose":r.purpose,"issued_go":r.issued_go,"requested":r.requested,"issued":r.issued,
                "readiness_barrier_completed":r.readiness_barrier_completed,"go_issued":true,"raw":raw_report,"opinions":opinions_json,
                "requested_multipv":outcome.requested_multipv,"conditional_policy_target_known":outcome.conditional_policy_target_known,
                "option_application_observed":r.option_application_observed,"model_loading_observed":startup.model_loading_observed(),"attempt":attempt_json,
                "stop_control":outcome.stop_control.map(|c|json!({"sequence":c.sequence,"action":format!("{:?}",c.action)})),
                "timing":{"go_issued":timing.go_issued_at.is_some(),"stop_sent":timing.stop_sent_at.is_some(),
                    "bestmove_drained":timing.bestmove_drained_at.is_some(),"ready_completed":timing.ready_completed_at.is_some()}}});
        let seal = generation_digest(HEAD_DOMAIN, &body)?;
        row.record.targets.policy = Some(GenerationPolicyTargetV3 {
            observed_sequence: sequence,
            question_id: row.record.input.question_id.clone(),
            view_sha256: row.record.input.view_sha256.clone(),
            raw_evidence_sha256: seal.clone(),
            source: "rovent_completed_teacher_multipv".into(),
            completed: true,
            stale: false,
            anchor: row.record.input.rules_anchor.clone(),
            ranked_moves: ranked,
            teacher_binary_sha256: pin.binary_sha256.clone(),
            teacher_profile_sha256: pin.profile_sha256.clone(),
            go_sequence: raw.request_id,
            completed_depth: u32::from(depth),
        });
        row.record.head_sources.insert(
            "policy".into(),
            RoventHeadSource {
                actor_id,
                source: "rovent_completed_teacher_multipv".into(),
                binary_sha256: Some(pin.binary_sha256),
                profile_sha256: Some(pin.profile_sha256),
                raw_receipt: body,
                raw_receipt_sha256: seal,
                observed_sequence: sequence,
                anchor: row.record.input.rules_anchor.clone(),
            },
        );
        Ok(())
    }
}
