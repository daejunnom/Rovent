//! 독립 Collection CPU_T와 원 submitted student input의 결합 초안이다.
//! 전용 TeacherJob을 사용하며 R QuestionEnvelope·수신자·native RequestId를 만들지 않는다.
//! 상위 teacher.rs의 private 완료 직렬화/target 적용 함수만 재사용한다.
use super::*;
use rz_search::external_cpu::collection_teacher::{
    TeacherCompletion, TeacherDispatch, TeacherInputScope, TeacherJob, TeacherJobId,
};
use rz_uci::rovent_external_launch::RoventCollectionTeacherOwner;
use std::sync::Arc;

pub const ROVENT_COLLECTION_TEACHER_BINDING: &str = "rz-rovent-collection-teacher-binding/1";
pub const ROVENT_FROZEN_TEACHER_ACTOR: &str = "rz-rovent-frozen-teacher-actor/1";
fn digest(value: &str) -> Result<Digest, ArenaError> {
    check(generation_sha(value), "teacher SHA syntax")?;
    let mut bytes = [0; 32];
    for (i, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        let digit = |b: u8| if b <= b'9' { b - b'0' } else { b - b'a' + 10 };
        bytes[i] = digit(pair[0]) * 16 + digit(pair[1]);
    }
    Ok(Digest(bytes))
}
pub(super) fn scope_json(scope: TeacherInputScope) -> Value {
    json!({"authority":{"process_epoch":scope.authority.epoch.0,"game_generation":scope.authority.game.0,
        "root_generation":scope.authority.root.0,"implementation_sha256":hex(scope.authority.implementation)},
        "input_sha256":hex(scope.input_sha256),"view_sha256":hex(scope.view_sha256),"source_sha256":hex(scope.source_sha256),
        "role":role_name(scope.role),"rules_semantic":hex(scope.rules_semantic),"input_lock":hex(scope.input_lock),
        "root_admission":hex(scope.root_admission),"root_budget":hex(scope.root_budget),"frozen_actor":hex(scope.frozen_actor)})
}
impl RoventRegisteredTeacher {
    /// 원 prepared/profile/pin 대조를 마친 등록 actor의 metadata seal이다.
    /// 이 값 자체는 startup·go·교사 label 권한이 아니다.
    pub fn frozen_actor_digest(&self) -> Result<Digest, ArenaError> {
        digest(&generation_digest(
            ROVENT_FROZEN_TEACHER_ACTOR,
            &json!({"actor_id":self.actor_id,"teacher_pin":self.pin}),
        )?)
    }
}
#[derive(Clone)]
pub struct RoventCollectionTeacherBinding {
    input_sha256: String,
    anchor: GenerationRulesAnchorV3,
    actor_id: String,
    pin: RoventTeacherPin,
    id: TeacherJobId,
    scope: TeacherInputScope,
    constraint: rz_search::external_cpu::ExternalGoConstraint,
    root_restriction: Digest,
    process: ExternalProcessIdentity,
    root_admission: Arc<rz_uci::rovent_root_admission::RoventRootAdmission>,
    raw: Value,
}
impl RoventCollectionTeacherBinding {
    pub fn input_sha256(&self) -> &str {
        &self.input_sha256
    }
}
impl RoventCorpusProducer {
    /// 실제 원 입력 seal의 read-only projection이다. 반환 metadata만으로
    /// worker 완료·학습 target을 만들 수 없고 아래 opaque binding이 별도 필요하다.
    pub fn collection_teacher_scope(
        &self,
        input: &RoventSubmittedInputSeal,
        registered: &RoventRegisteredTeacher,
        owner: &RoventCollectionTeacherOwner,
    ) -> Result<TeacherInputScope, ArenaError> {
        let root = owner.root_admission();
        check(
            input.manifest_sha256 == self.manifest_sha256
                && input.teacher_actor_id.as_ref() == Some(&registered.actor_id)
                && self.manifest.teachers.get(&registered.actor_id) == Some(&registered.pin)
                && root.authority() == input.ticket.authority
                && root.hard_deadline() == input.ticket.original_deadline
                && owner.frozen_actor() == registered.frozen_actor_digest()?,
            "Collection teacher original submitted input/actor/root",
        )?;
        let role = match input.role {
            GenerationRoleV3::Proposer => Role::Proposer,
            GenerationRoleV3::Critic => Role::Critic,
            GenerationRoleV3::Validator => {
                return Err(invalid("V-free Collection teacher accepts only P/C"));
            }
        };
        Ok(TeacherInputScope {
            authority: input.ticket.authority,
            input_sha256: digest(&input.input_sha256)?,
            // This is GenerationInputV3.view_sha256, not native ticket.view.view_digest.
            view_sha256: digest(&input.selected_view_sha256)?,
            source_sha256: digest(&self.manifest.source_sha256)?,
            role,
            rules_semantic: digest(&input.anchor.rules_state_sha256)?,
            input_lock: root.startup_lock_digest(),
            root_admission: root.digest(),
            root_budget: root.root_budget_digest(),
            frozen_actor: registered.frozen_actor_digest()?,
        })
    }
    /// 원 제출 입력과 실제 teacher startup을 go 발행 전에 고정한다. 질문의
    /// Position을 이 함수가 변경하지 않으며 미래 의견은 입력에 삽입하지 않는다.
    pub fn bind_collection_teacher_input(
        &self,
        input: &RoventSubmittedInputSeal,
        registered: &RoventRegisteredTeacher,
        owner: &RoventCollectionTeacherOwner,
        job: &TeacherJob,
    ) -> Result<RoventCollectionTeacherBinding, ArenaError> {
        let scope = self.collection_teacher_scope(input, registered, owner)?;
        let startup = owner
            .verified_startup()
            .map_err(|e| invalid(e.to_string()))?
            .ok_or_else(|| invalid("Collection teacher actual startup pending"))?;
        let root = owner.root_admission();
        let registration = owner.worker().registration();
        let anchor = exact_rules_anchor(&job.position)?;
        check(
            job.input_scope == scope
                && job.id.collection_owner == registration.collection_owner
                && job.id.sequence > 0
                && anchor == input.anchor
                && job.original_deadline.clock == root.hard_deadline().clock
                && job.original_deadline.at <= root.hard_deadline().at
                && startup.canonical_sha256() == registered.pin.canonical_sha256
                && startup.capability_sha256() == registered.pin.capability_sha256
                && startup.input_lock_sha256() == hex(root.startup_lock_digest())
                && owner.launch().binary().sha256 == registered.pin.binary_sha256
                && owner.launch().binary().bytes == registered.pin.binary_bytes,
            "Collection teacher Job/startup differs from original input/actor",
        )?;
        let root_restriction = rz_search::external_cpu::worker::external_root_restriction_digest(
            job.root_moves.as_deref(),
        );
        let raw = json!({"schema":ROVENT_COLLECTION_TEACHER_BINDING,
            "job_id":{"collection_owner":job.id.collection_owner,"sequence":job.id.sequence},
            "input_scope":scope_json(scope),"requested":job.constraint,"process":process_json(startup.process_identity()),
            "root_restriction":hex(root_restriction),"position_anchor":anchor,
            "root_moves":job.root_moves.as_ref().map(|v|packed(v)).transpose()?,
            "soft_quantum_ns":job.soft_quantum.as_nanos().min(u128::from(u64::MAX)) as u64,
            "original_deadline":{"clock_epoch":job.original_deadline.clock.0.0,"at_ns":job.original_deadline.at.0},
            "root_descriptor":root.descriptor(),"root_descriptor_sha256":hex(root.digest())});
        Ok(RoventCollectionTeacherBinding {
            input_sha256: input.input_sha256.clone(),
            anchor,
            actor_id: registered.actor_id.clone(),
            pin: registered.pin.clone(),
            id: job.id,
            scope,
            constraint: job.constraint,
            root_restriction,
            process: startup.process_identity(),
            root_admission: Arc::clone(root),
            raw,
        })
    }
    /// 오직 같은 원 private dispatch/completion으로 완료된 합법 MultiPV
    /// 의견을 policy로 사용한다. CP·mate·엔진 동의는 WDL target이 아니다.
    pub fn attach_collection_teacher_policy(
        &mut self,
        row: &mut RoventRetiredRecord,
        binding: RoventCollectionTeacherBinding,
        owner: &RoventCollectionTeacherOwner,
        dispatch: &TeacherDispatch,
        completion: &TeacherCompletion,
    ) -> Result<(), ArenaError> {
        owner
            .validate_completion(completion, dispatch)
            .map_err(|e| invalid(e.to_string()))?;
        check(
            row.record.targets.policy.is_none()
                && binding.input_sha256 == row.record.input_sha256
                && binding.anchor == row.record.input.rules_anchor
                && Arc::ptr_eq(&binding.root_admission, row.admission.root_admission())
                && Arc::ptr_eq(&binding.root_admission, owner.root_admission())
                && dispatch.id() == binding.id
                && dispatch.input_scope() == binding.scope
                && dispatch.constraint() == binding.constraint
                && dispatch.root_restriction() == binding.root_restriction
                && dispatch.process_identity() == binding.process
                && completion.go_issue() == ExternalGoIssue::Issued,
            "Collection teacher original private dispatch/input mismatch",
        )?;
        let startup = owner
            .verified_startup()
            .map_err(|e| invalid(e.to_string()))?
            .ok_or_else(|| invalid("Collection teacher original startup absent"))?;
        let outcome = completion
            .result()
            .as_ref()
            .map_err(|e| invalid(e.to_string()))?;
        let attempt = completion
            .attempt()
            .ok_or_else(|| invalid("Collection teacher original physical attempt absent"))?;
        // Next registered window extracts this private helper from the existing
        // legacy method unchanged. Both methods first validate their own opaque
        // physical completion; the helper is not a public JSON authority API.
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
}
