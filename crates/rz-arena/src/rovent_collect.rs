//! Rovent actual-input producer. The wire reader is deliberately not an
//! execution authority: rows originate in the private native owner witness and
//! the registry's AcceptedRoleStep, not deserialized success flags.
use crate::ArenaError;
use crate::pals_collect_v3::{
    ActualGameTrajectoryV3, GenerationCaptureContextV3, capture_submitted_review_v3,
    exact_rules_anchor,
};
use rz_contracts::pals::{Role, RoleOutput, RolePayload};
use rz_contracts::pals_adaptive::RoleTicketEnvelope;
use rz_contracts::{Digest, RequestId};
use rz_eval::pals_model::virtual_review::{PalsVirtualReviewInput, ReviewExecutionIdentity};
use rz_experiments::pals_data_v3::*;
use rz_experiments::rovent_corpus::*;
use rz_position::{BoardMove, Position};
use rz_search::pals::coordinator::{
    AcceptedRoleStep, PhysicalRoleCompletion, ProposerRankingOutput,
    native_acceptance_output_digest,
};
use rz_search::pals::review_context::ReviewView;
use rz_search::pals::store::Move16;
use rz_uci::pals_adaptive::{RoventCollectionRootAdmission, RoventRootRankingAcceptance};
use rz_uci::pals_native::ticketed::TicketNativeEvent;
use serde_json::{Value, json};
use sha2::{Digest as _, Sha256};
use std::collections::{BTreeMap, BTreeSet};

fn invalid(message: impl Into<String>) -> ArenaError {
    ArenaError::Invalid(message.into())
}
fn check(ok: bool, message: &str) -> Result<(), ArenaError> {
    if ok { Ok(()) } else { Err(invalid(message)) }
}
fn hex(value: Digest) -> String {
    value.0.iter().map(|v| format!("{v:02x}")).collect()
}
fn packed(moves: &[BoardMove]) -> Result<Vec<u16>, ArenaError> {
    moves
        .iter()
        .map(|m| {
            Move16::pack(*m)
                .map(|m| m.bits())
                .map_err(|e| invalid(e.to_string()))
        })
        .collect()
}
fn role_name(role: Role) -> &'static str {
    match role {
        Role::Proposer => "proposer",
        Role::Critic => "critic",
        Role::Validator => "validator",
    }
}
fn ticket_json(t: &RoleTicketEnvelope) -> Value {
    json!({"process_epoch":t.id.epoch.0,"game_generation":t.authority.game.0,
        "root_generation":t.authority.root.0,"request_sequence":t.id.sequence,
        "implementation_sha256":hex(t.authority.implementation),"question_digest":hex(t.question),
        "view_sequence":t.view.view_sequence,"view_digest":hex(t.view.view_digest),
        "context_policy":hex(t.view.context_policy),"actual_tokens":t.view.actual_tokens,
        "selected_records":t.view.selected_records,"anchors":t.view.anchors,"length_bucket":t.view.length_bucket,
        "root_state":{"owner":t.view.root_state.owner.0,"revision":t.view.root_state.revision.0,"semantic":hex(t.view.root_state.semantic)},
        "representation":{"input":hex(t.representation.input),"history":hex(t.representation.history),
            "candidates":hex(t.representation.candidates),"mask_positions":hex(t.representation.mask_positions),
            "model":hex(t.representation.model),"encoding":hex(t.representation.encoding),
            "precision":format!("{:?}",t.representation.precision),"frozen_epoch":t.representation.frozen_epoch,
            "record_revision":t.representation.record_revision,"role":role_name(t.role)},
        "deadline_clock":t.original_deadline.clock.0.0,"deadline_tick":t.original_deadline.at.0})
}
/// Lossless f32 bits are part of the raw owner body. They avoid a second,
/// language-dependent JSON float spelling in corpus semantic digests.
fn output_json(o: &RoleOutput) -> Value {
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let payload = match &o.payload {
        RolePayload::Proposal {
            candidate_logits,
            wdl,
        } => {
            json!({"kind":"proposal","candidate_logits_bits":bits(candidate_logits),"wdl_bits":bits(&wdl.probabilities())})
        }
        RolePayload::Counterexample {
            candidate_logits,
            divergence_logits,
            wdl,
        } => {
            json!({"kind":"counterexample","candidate_logits_bits":bits(candidate_logits),"divergence_logits_bits":bits(divergence_logits),"wdl_bits":bits(&wdl.probabilities())})
        }
        RolePayload::TaskRanking { task_logits } => {
            json!({"kind":"task_ranking","task_logits_bits":bits(task_logits)})
        }
    };
    json!({"request_epoch":o.id.epoch.0,"request_sequence":o.id.sequence,
        "game_generation":o.authority.game.0,"root_generation":o.authority.root.0,
        "implementation_sha256":hex(o.authority.implementation),
        "representation":{"input":hex(o.key.input),"history":hex(o.key.history),"candidates":hex(o.key.candidates),
            "mask_positions":hex(o.key.mask_positions),"model":hex(o.key.model),"encoding":hex(o.key.encoding),
            "precision":format!("{:?}",o.key.precision),"frozen_epoch":o.key.frozen_epoch,"record_revision":o.key.record_revision,"role":role_name(o.key.role)},
        "payload":payload,"private_latent_bits":bits(&o.private_latent)})
}
struct Pending {
    ticket: RoleTicketEnvelope,
    input: GenerationInputV3,
    input_sha256: String,
    lineage: RoventLineage,
    submitted: u64,
    raw_submission: Value,
    completed: Option<RoventNativeLifecycle>,
}
/// manager가 예약한 원 문맥이다. 입력 인코딩은 이 값을 받은 native owner가
/// 수행하며, 공개 JSON으로 이 예약이나 제출 증거를 복원할 수 없다.
pub struct RoventInputCaptureReservation {
    manifest_sha256: String,
    source_sha256: String,
    context: GenerationCaptureContextV3,
    lineage: RoventLineage,
}
pub struct RoventCapturedInput {
    manifest_sha256: String,
    pending: Pending,
}
/// 제출 입력을 교사 질문에 연결하는 권한이다. NN 실행·완료·수락이나
/// 학습 가능한 record를 의미하지 않으며, future target은 포함하지 않는다.
pub struct RoventSubmittedInputSeal {
    manifest_sha256: String,
    ticket: RoleTicketEnvelope,
    input_sha256: String,
    // GenerationInputV3 selected-view seal, distinct from native ticket view digest.
    selected_view_sha256: String,
    anchor: GenerationRulesAnchorV3,
    role: GenerationRoleV3,
    teacher_actor_id: Option<String>,
}
impl RoventCapturedInput {
    pub fn submitted_input_seal(&self) -> RoventSubmittedInputSeal {
        input_seal(
            &self.manifest_sha256,
            self.pending.ticket,
            &self.pending.input,
            &self.pending.input_sha256,
            &self.pending.lineage,
        )
    }
}
fn input_seal(
    manifest: &str,
    ticket: RoleTicketEnvelope,
    input: &GenerationInputV3,
    input_sha256: &str,
    lineage: &RoventLineage,
) -> RoventSubmittedInputSeal {
    RoventSubmittedInputSeal {
        manifest_sha256: manifest.into(),
        ticket,
        input_sha256: input_sha256.into(),
        selected_view_sha256: input.view_sha256.clone(),
        anchor: input.rules_anchor.clone(),
        role: input.role,
        teacher_actor_id: lineage.teacher_actor_id.clone(),
    }
}
impl RoventInputCaptureReservation {
    pub fn sequence(&self) -> u64 {
        self.context.capture_sequence
    }
    pub fn capture(
        self,
        ticket: RoleTicketEnvelope,
        view: &ReviewView,
        input: &PalsVirtualReviewInput,
        identity: &ReviewExecutionIdentity,
    ) -> Result<RoventCapturedInput, ArenaError> {
        input.validate().map_err(|e| invalid(e.to_string()))?;
        check(
            input
                .fresh_key(identity)
                .map_err(|e| invalid(e.to_string()))?
                .0
                == ticket.representation.input.0,
            "captured Rust input differs from original native representation",
        )?;
        let captured = capture_submitted_review_v3(&ticket, view, input, &self.context)?;
        let input_sha256 = captured.sha256()?;
        let replay =
            rules_replay::RoventSubmittedRulesReplay::capture(&self.context.actual_position, view)?;
        let raw = json!({"schema":NATIVE_RAW_RULES_V2_DOMAIN,"kind":"submitted","sequence":self.context.capture_sequence,
            "process_epoch":ticket.id.epoch.0,"game_generation":ticket.authority.game.0,"root_generation":ticket.authority.root.0,
            "request_sequence":ticket.id.sequence,"head":role_name(ticket.role),"source_sha256":self.source_sha256,
            "model_sha256":hex(ticket.representation.model),"encoding_sha256":hex(ticket.representation.encoding),
            "input_digest":hex(ticket.representation.input),"view_digest":hex(ticket.view.view_digest),
            "input_sha256":input_sha256,"ticket":ticket_json(&ticket),"rules_replay":replay});
        Ok(RoventCapturedInput {
            manifest_sha256: self.manifest_sha256,
            pending: Pending {
                ticket,
                input: captured,
                input_sha256,
                lineage: self.lineage,
                submitted: self.context.capture_sequence,
                raw_submission: raw,
                completed: None,
            },
        })
    }
}
/// Private construction makes it impossible for wire metadata to create an
/// actual retired record. Targets may be attached later, without changing input.
pub struct RoventRetiredRecord {
    record: RoventTrainingRecord,
    manifest_sha256: String,
    ticket: RoleTicketEnvelope,
    admission: RoventCollectionRootAdmission,
}
impl RoventRetiredRecord {
    pub fn input(&self) -> &GenerationInputV3 {
        &self.record.input
    }
    pub fn input_sha256(&self) -> &str {
        &self.record.input_sha256
    }
    pub fn submitted_input_seal(&self) -> RoventSubmittedInputSeal {
        input_seal(
            &self.manifest_sha256,
            self.ticket,
            &self.record.input,
            &self.record.input_sha256,
            &self.record.lineage,
        )
    }
}
pub struct PublishedRoventRecord {
    record: RoventTrainingRecord,
    manifest_sha256: String,
    admission: RoventCollectionRootAdmission,
}
impl PublishedRoventRecord {
    pub fn record(&self) -> &RoventTrainingRecord {
        &self.record
    }
    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }
    pub fn admission(&self) -> &RoventCollectionRootAdmission {
        &self.admission
    }
}
/// Single-writer collector. It neither runs engines nor initiates collection.
/// Its local sequence is independent of Store revisions/native request IDs.
pub struct RoventCorpusProducer {
    native_domain: &'static str,
    manifest: RoventCorpusManifest,
    manifest_sha256: String,
    sequence: u64,
    reserved: BTreeSet<u64>,
    pending: BTreeMap<(u64, u64), Pending>,
    retired: BTreeSet<(u64, u64)>,
    native_observation_failures:
        BTreeMap<RequestId, Vec<rz_uci::pals_native::ticketed::TicketNativeObservationFailure>>,
}
impl RoventCorpusProducer {
    fn validate_admission(
        &self,
        admission: &RoventCollectionRootAdmission,
    ) -> Result<(), ArenaError> {
        let original = admission.native_origin().raw_receipt();
        check(
            original["source_commit"] == self.manifest.source_commit
                && original["source_sha256"] == self.manifest.source_sha256
                && original["runtime_sha256_domain"] == "loaded_runtime_library_file_sha256"
                && original["runtime_library_sha256"] == self.manifest.runtime_sha256
                && self.manifest.model_semantics_sha256
                    == hex(Digest(
                        rz_eval::pals_model::PalsModelProfile::VirtualReviewV1
                            .registered_semantics_sha256(),
                    )),
            "corpus manifest does not join the original loaded Rovent admission",
        )
    }
    pub fn new(manifest: RoventCorpusManifest) -> Result<Self, ArenaError> {
        let manifest_sha256 = manifest.sha256()?;
        Ok(Self {
            native_domain: NATIVE_RAW_RULES_V2_DOMAIN,
            manifest,
            manifest_sha256,
            sequence: 0,
            reserved: BTreeSet::new(),
            pending: BTreeMap::new(),
            retired: BTreeSet::new(),
            native_observation_failures: BTreeMap::new(),
        })
    }
    pub fn manifest(&self) -> &RoventCorpusManifest {
        &self.manifest
    }
    pub fn incomplete_capture_count(&self) -> usize {
        self.reserved.len() + self.pending.len()
    }
    pub fn native_observation_failures(
        &self,
        id: RequestId,
    ) -> Option<&[rz_uci::pals_native::ticketed::TicketNativeObservationFailure]> {
        self.native_observation_failures.get(&id).map(Vec::as_slice)
    }
    pub fn submitted_input_seal(
        &self,
        id: RequestId,
    ) -> Result<RoventSubmittedInputSeal, ArenaError> {
        let p = self
            .pending
            .get(&(id.epoch.0, id.sequence))
            .ok_or_else(|| invalid("original submitted input is not available"))?;
        Ok(input_seal(
            &self.manifest_sha256,
            p.ticket,
            &p.input,
            &p.input_sha256,
            &p.lineage,
        ))
    }
    fn next(&mut self) -> Result<u64, ArenaError> {
        self.sequence = self
            .sequence
            .checked_add(1)
            .ok_or_else(|| invalid("Rovent collector sequence exhausted"))?;
        Ok(self.sequence)
    }
    pub fn capture_submission(
        &mut self,
        ticket: RoleTicketEnvelope,
        view: &ReviewView,
        input: &PalsVirtualReviewInput,
        identity: &ReviewExecutionIdentity,
        context: GenerationCaptureContextV3,
        lineage: RoventLineage,
    ) -> Result<(), ArenaError> {
        let reserved = self.reserve_input(context, lineage)?;
        let sequence = reserved.sequence();
        let result = reserved
            .capture(ticket, view, input, identity)
            .and_then(|captured| self.admit_captured_input(captured));
        if result.is_err() {
            self.reserved.remove(&sequence);
        }
        result
    }
    pub fn reserve_input(
        &mut self,
        mut context: GenerationCaptureContextV3,
        lineage: RoventLineage,
    ) -> Result<RoventInputCaptureReservation, ArenaError> {
        check(
            self.pending.len() + self.reserved.len() < 16,
            "over-cap native capture reservation",
        )?;
        check(
            context.model_epoch == self.manifest.model_epoch
                && context.context_epoch == self.manifest.context_epoch
                && generation_identifier(&lineage.actor_id)
                && lineage
                    .teacher_actor_id
                    .as_ref()
                    .is_none_or(|a| self.manifest.teachers.contains_key(a)),
            "native capture epoch/actor not registered",
        )?;
        context.capture_sequence = self.next()?;
        self.reserved.insert(context.capture_sequence);
        Ok(RoventInputCaptureReservation {
            manifest_sha256: self.manifest_sha256.clone(),
            source_sha256: self.manifest.source_sha256.clone(),
            context,
            lineage,
        })
    }
    /// native owner가 검증·인코딩한 값은 여기서 이동만 한다. manager는
    /// 전체 record를 재인코딩하거나 다시 해시하지 않는다.
    pub fn admit_captured_input(
        &mut self,
        captured: RoventCapturedInput,
    ) -> Result<(), ArenaError> {
        let key = (
            captured.pending.ticket.id.epoch.0,
            captured.pending.ticket.id.sequence,
        );
        check(
            captured.manifest_sha256 == self.manifest_sha256
                && self.reserved.contains(&captured.pending.submitted)
                && self.pending.len() < 16
                && !self.pending.contains_key(&key)
                && !self.retired.contains(&key),
            "revoked/different-owner/duplicate captured native input",
        )?;
        self.reserved.remove(&captured.pending.submitted);
        self.pending.insert(key, captured.pending);
        Ok(())
    }
    /// 취소한 입력 예약의 메타데이터만 반환한다. 물리 작업의 취소·drain은
    /// 원 native owner의 의무이며 이 함수로 완료 처리되지 않는다.
    pub fn discard_reserved_input(&mut self, sequence: u64) -> bool {
        self.reserved.remove(&sequence)
    }
    /// Consumes only the service's original private owner witness. A caller's
    /// public RoleCompletion or boolean is insufficient to reach this method.
    pub fn observe_native_event<O: ProposerRankingOutput>(
        &mut self,
        event: &TicketNativeEvent<O>,
    ) -> Result<(), ArenaError> {
        let terminal = match event {
            TicketNativeEvent::Completed {
                completion,
                observation_failures,
                ..
            } => Some((completion.ticket, observation_failures)),
            TicketNativeEvent::Retired {
                ticket,
                observation_failures,
                ..
            } => Some((*ticket, observation_failures)),
            _ => None,
        };
        if let Some((ticket, failures)) = terminal {
            if !failures.is_empty() {
                check(
                    self.native_observation_failures.len() < MAX_CORPUS_ROWS
                        || self.native_observation_failures.contains_key(&ticket.id),
                    "native observation failure evidence cap",
                )?;
                let retained = self
                    .native_observation_failures
                    .entry(ticket.id)
                    .or_default();
                retained.extend(failures.iter().cloned());
                return Err(invalid(format!(
                    "original native observation failed: {retained:?}"
                )));
            }
        }
        if let TicketNativeEvent::Retired {
            ticket,
            accepted: false,
            ..
        } = event
        {
            let key = (ticket.id.epoch.0, ticket.id.sequence);
            if let Some(p) = self.pending.get(&key) {
                check(
                    p.ticket == *ticket,
                    "rejected retirement belongs to a different frozen input",
                )?;
            }
            // 버리는 것은 수집용 입력 사본뿐이다. 원 owner의 물리 정리나
            // quarantine 상태를 완료로 바꾸지 않으며 record도 발급하지 않는다.
            self.pending.remove(&key);
            return Ok(());
        }
        let TicketNativeEvent::Completed {
            completion,
            evidence,
            owner_run,
            owner_service_elapsed,
            ..
        } = event
        else {
            return Ok(());
        };
        if matches!(
            &completion.result,
            Err(rz_search::pals::coordinator::RoleFailure::Canceled
                | rz_search::pals::coordinator::RoleFailure::Expired)
        ) {
            return Ok(());
        }
        let t = completion.ticket;
        let key = (t.id.epoch.0, t.id.sequence);
        let pending = self
            .pending
            .get(&key)
            .ok_or_else(|| invalid("completion has no submitted frozen input"))?;
        let e = evidence
            .as_ref()
            .ok_or_else(|| invalid("native completed without execution evidence"))?;
        let execution = e
            .execution
            .ok_or_else(|| invalid("native physical execution id unknown"))?;
        let output = completion
            .result
            .as_ref()
            .map_err(|e| invalid(format!("native unsuccessful completion: {e:?}")))?
            .proposer_ranking_output();
        check(
            pending.ticket == t
                && pending.completed.is_none()
                && owner_run.ticket() == &t
                && owner_run.backend_run_invoked()
                && completion.physical == PhysicalRoleCompletion::Confirmed
                && e.measurement_complete
                && e.physical_fresh == Some(true)
                && e.completed_nn_inputs.is_some_and(|n| n > 0)
                && execution.epoch == t.id.epoch
                && output.id == t.id
                && output.authority == t.authority
                && output.key == t.representation,
            "native witness/completion/fresh/identity mismatch",
        )?;
        let output_digest =
            hex(native_acceptance_output_digest(output).map_err(|e| invalid(format!("{e:?}")))?);
        let completed = self.next()?;
        let pending = self.pending.get_mut(&key).expect("checked pending");
        let submission_sha = generation_digest(self.native_domain, &pending.raw_submission)?;
        let raw = json!({"schema":self.native_domain,"kind":"completed","sequence":completed,
            "process_epoch":t.id.epoch.0,"game_generation":t.authority.game.0,"root_generation":t.authority.root.0,
            "request_sequence":t.id.sequence,"source_sha256":self.manifest.source_sha256,"head":role_name(t.role),
            "input_sha256":pending.input_sha256,"submitted_sha256":submission_sha,
            "execution_epoch":execution.epoch.0,"execution_sequence":execution.sequence,
            "completed_nn_inputs":e.completed_nn_inputs,"physical_confirmed":true,"owner_run_invoked":true,
            "measurement_complete":e.measurement_complete,"physical_fresh":e.physical_fresh,
            "output_digest":output_digest,"owner_service_elapsed_ns":owner_service_elapsed.map(|d|d.as_nanos().min(u128::from(u64::MAX)) as u64),
            "ticket":ticket_json(&t),"output":output_json(output)});
        pending.completed = Some(RoventNativeLifecycle {
            head: role_name(t.role).into(),
            source_sha256: self.manifest.source_sha256.clone(),
            model_sha256: hex(t.representation.model),
            encoding_sha256: hex(t.representation.encoding),
            input_digest: hex(t.representation.input),
            view_digest: hex(t.view.view_digest),
            output_digest,
            process_epoch: t.id.epoch.0,
            game_generation: t.authority.game.0,
            root_generation: t.authority.root.0,
            request_sequence: t.id.sequence,
            execution_epoch: execution.epoch.0,
            execution_sequence: execution.sequence,
            completed_nn_inputs: e.completed_nn_inputs.expect("checked observed"),
            submitted_sequence: pending.submitted,
            completed_sequence: completed,
            retired_sequence: 0,
            owner_run_invoked: true,
            physical_confirmed: true,
            accepted_retired: false,
            raw_submission: pending.raw_submission.clone(),
            raw_submission_sha256: submission_sha,
            raw_completion_sha256: generation_digest(self.native_domain, &raw)?,
            raw_completion: raw,
            raw_retirement: Value::Null,
            raw_retirement_sha256: String::new(),
        });
        Ok(())
    }
    /// AcceptedRoleStep is minted only after the native owner ack and registry
    /// retirement. Rejected/late/quarantined events never publish a training row.
    pub fn retire_accepted(
        &mut self,
        admission: &RoventCollectionRootAdmission,
        accepted: &AcceptedRoleStep,
        output: &RoleOutput,
    ) -> Result<RoventRetiredRecord, ArenaError> {
        let t = *accepted.ticket();
        admission
            .validate_output(&t, output)
            .map_err(|error| invalid(error.to_string()))?;
        check(
            accepted.validates_output(output),
            "original accepted step/output mismatch",
        )?;
        self.retire_original(admission, t, accepted.output_digest(), "accepted_role_step")
    }
    /// PCR 최초 P의 원 publication을 별도 실제 proof로 소비한다. 이를
    /// reciprocal AcceptedRoleStep으로 재발급하거나 변환하지 않는다.
    pub fn retire_root_ranking(
        &mut self,
        accepted: &RoventRootRankingAcceptance,
        output: &RoleOutput,
    ) -> Result<RoventRetiredRecord, ArenaError> {
        let ranking = accepted.ranking();
        let source = ranking.source();
        accepted
            .admission()
            .validate_output(&source.ticket, output)
            .map_err(|error| invalid(error.to_string()))?;
        check(
            native_acceptance_output_digest(output).map_err(|error| invalid(error.to_string()))?
                == source.output_digest,
            "original P publication/output digest mismatch",
        )?;
        self.retire_original(
            accepted.admission(),
            source.ticket,
            source.output_digest,
            "root_ranking_published",
        )
    }
    fn retire_original(
        &mut self,
        admission: &RoventCollectionRootAdmission,
        t: RoleTicketEnvelope,
        output_digest: Digest,
        acceptance_kind: &str,
    ) -> Result<RoventRetiredRecord, ArenaError> {
        self.validate_admission(admission)?;
        check(
            !self.native_observation_failures.contains_key(&t.id),
            "native observation failed for original retirement",
        )?;
        let key = (t.id.epoch.0, t.id.sequence);
        let pending = self
            .pending
            .get(&key)
            .ok_or_else(|| invalid("retirement has no original input"))?;
        let completed = pending
            .completed
            .as_ref()
            .ok_or_else(|| invalid("retirement before physical completion"))?;
        check(
            pending.ticket == t && completed.output_digest == hex(output_digest),
            "registry retirement/output does not match owner completion",
        )?;
        check(
            self.retired.len() < MAX_CORPUS_ROWS,
            "retired ticket deduplication cap",
        )?;
        let retired = self.next()?;
        let pending = self.pending.remove(&key).expect("checked pending");
        let mut native = pending.completed.expect("checked completed");
        let raw = json!({"schema":self.native_domain,"kind":"retired","sequence":retired,
            "process_epoch":t.id.epoch.0,"game_generation":t.authority.game.0,"root_generation":t.authority.root.0,
            "request_sequence":t.id.sequence,"head":role_name(t.role),"source_sha256":self.manifest.source_sha256,
            "input_sha256":pending.input_sha256,"submitted_sha256":native.raw_submission_sha256,
            "completed_sha256":native.raw_completion_sha256,"output_digest":native.output_digest,
            "accepted_retired":true,"ticket":ticket_json(&t),"acceptance_kind":acceptance_kind,
            "root_admission":admission.raw_receipt(),"root_admission_sha256":admission.raw_sha256()});
        native.raw_retirement_sha256 = generation_digest(self.native_domain, &raw)?;
        native.raw_retirement = raw;
        native.retired_sequence = retired;
        native.accepted_retired = true;
        self.retired.insert(key);
        let split = *self
            .manifest
            .opening_splits
            .get(&pending.input.opening_id)
            .ok_or_else(|| invalid("unregistered opening split"))?;
        Ok(RoventRetiredRecord {
            manifest_sha256: self.manifest_sha256.clone(),
            ticket: t,
            admission: admission.clone(),
            record: RoventTrainingRecord {
                schema: RECORD_DOMAIN.into(),
                role: pending.input.role,
                split,
                input_sha256: pending.input_sha256,
                input: pending.input,
                origin_sha256: String::new(),
                record_sha256: String::new(),
                lineage: pending.lineage,
                native,
                head_sources: BTreeMap::new(),
                targets: GenerationTargetsV3::default(),
                divergence_index: None,
                nnuec_input: None,
                nnuec_input_sha256: None,
                nnuec_model_sha256: None,
                nnuec_semantics: None,
                nnuec_wdl: None,
            },
        })
    }
    /// Explicit retirement of rejected work only removes its frozen input.
    /// The caller still owns the original physical cancel/drain obligation.
    pub fn discard_unpublished(&mut self, id: RequestId) -> bool {
        self.pending.remove(&(id.epoch.0, id.sequence)).is_some()
    }
    pub fn publish(
        &self,
        mut row: RoventRetiredRecord,
    ) -> Result<PublishedRoventRecord, ArenaError> {
        check(
            row.manifest_sha256 == self.manifest_sha256,
            "retired row belongs to a different corpus/stage owner",
        )?;
        self.validate_admission(&row.admission)?;
        check(
            row.record.native.raw_retirement["root_admission_sha256"] == row.admission.raw_sha256()
                && row.record.native.raw_retirement["root_admission"]
                    == *row.admission.raw_receipt(),
            "retired row lost its original Rovent root admission",
        )?;
        row.record.seal_rules_replay_v2(&self.manifest)?;
        Ok(PublishedRoventRecord {
            record: row.record,
            manifest_sha256: row.manifest_sha256,
            admission: row.admission,
        })
    }
    pub fn attach_rust_nnuec(
        &self,
        _row: &mut RoventRetiredRecord,
        _input: RustNnuecReference,
    ) -> Result<(), ArenaError> {
        Err(invalid(
            "MissingOriginalNnuecCapture: research numeric reference is not an original submitted CPU Job/readout/retirement capability",
        ))
    }
    pub fn attach_actual_wdl(
        &mut self,
        row: &mut RoventRetiredRecord,
        outcome: &RoventRulesOutcome,
    ) -> Result<(), ArenaError> {
        let input = &row.record.input;
        check(
            row.record.targets.wdl.is_none()
                && outcome.authority.game_id == input.game_id
                && outcome.authority.anchors.contains(&input.rules_anchor),
            "WDL input is not an actual member of the completed game",
        )?;
        let sequence = self.next()?;
        let raw = json!({"schema":HEAD_DOMAIN,"kind":"rules_wdl","input_sha256":row.record.input_sha256,
            "anchor":input.rules_anchor,"trajectory":outcome.authority,"rules_receipt":outcome.receipt,
            "rules_replay":outcome.replay});
        let raw_sha = generation_digest(HEAD_DOMAIN, &raw)?;
        let target = GenerationWdlTargetV3 {
            observed_sequence: sequence,
            question_id: input.question_id.clone(),
            view_sha256: input.view_sha256.clone(),
            raw_evidence_sha256: raw_sha.clone(),
            source: "rules_selfplay_trajectory".into(),
            game_id: input.game_id.clone(),
            anchor: input.rules_anchor.clone(),
            trajectory_membership_sha256: outcome.authority.sha256()?,
            rules_result_sha256: outcome.authority.result_evidence_sha256.clone(),
            ending: outcome.authority.ending,
            outcome: outcome.authority.outcome,
        };
        target.probabilities(input)?;
        let source = RoventHeadSource {
            actor_id: row.record.lineage.actor_id.clone(),
            source: target.source.clone(),
            binary_sha256: None,
            profile_sha256: None,
            raw_receipt: raw,
            raw_receipt_sha256: raw_sha,
            observed_sequence: sequence,
            anchor: input.rules_anchor.clone(),
        };
        row.record.targets.wdl = Some(target.clone());
        row.record.head_sources.insert("wdl".into(), source);
        Ok(())
    }
}

/// No Deserialize/public constructor: only Rules-owned actual move playback
/// can create the outcome used above. A virtual endpoint cannot issue it.
pub struct RoventRulesOutcome {
    authority: GenerationTrajectoryAuthorityV3,
    receipt: Value,
    replay: RoventActualTrajectoryReplay,
}
pub struct RoventGameTrajectory {
    inner: ActualGameTrajectoryV3,
    initial: RoventRulesPositionReplay,
    selected_moves: Vec<u16>,
}
impl RoventGameTrajectory {
    pub fn begin(game_id: String, initial: Position) -> Result<Self, ArenaError> {
        let replay = RoventRulesPositionReplay::capture(&initial)?;
        Ok(Self {
            inner: ActualGameTrajectoryV3::begin(game_id, initial)?,
            initial: replay,
            selected_moves: Vec::new(),
        })
    }
    pub fn position(&self) -> &Position {
        self.inner.position()
    }
    pub fn play_selected_move(&mut self, movement: BoardMove) -> Result<(), ArenaError> {
        let packed = Move16::pack(movement)
            .map_err(|error| invalid(error.to_string()))?
            .bits();
        self.inner.play_pals_move(movement)?;
        self.selected_moves.push(packed);
        Ok(())
    }
    pub fn finish_rules(&mut self) -> Result<RoventRulesOutcome, ArenaError> {
        let replay = self.outcome_replay(RoventRulesCompletion::Automatic, None)?;
        let (authority, receipt) = self.inner.terminate_rules()?;
        Ok(RoventRulesOutcome {
            authority,
            receipt,
            replay,
        })
    }
    pub fn claim_rules_draw(
        &mut self,
        reason: rz_position::ClaimReason,
    ) -> Result<RoventRulesOutcome, ArenaError> {
        let claim = match reason {
            rz_position::ClaimReason::ThreefoldRepetition => RoventRulesClaim::ThreefoldRepetition,
            rz_position::ClaimReason::FiftyMove => RoventRulesClaim::FiftyMove,
        };
        let replay = self.outcome_replay(RoventRulesCompletion::CurrentClaim, Some(claim))?;
        let (authority, receipt) = self.inner.claim_rules_draw(reason)?;
        Ok(RoventRulesOutcome {
            authority,
            receipt,
            replay,
        })
    }
    fn outcome_replay(
        &self,
        completion: RoventRulesCompletion,
        claim: Option<RoventRulesClaim>,
    ) -> Result<RoventActualTrajectoryReplay, ArenaError> {
        check(
            self.selected_moves.len() <= 512,
            "actual Rules trajectory ply capacity",
        )?;
        Ok(RoventActualTrajectoryReplay {
            schema: ROVENT_ACTUAL_TRAJECTORY_REPLAY.into(),
            initial: self.initial.clone(),
            actual_selected_moves: self.selected_moves.clone(),
            final_position: RoventRulesPositionReplay::capture(self.inner.position())?,
            completion,
            claim,
        })
    }
}
/// 연구용 수치 reference다. 원 CPU 질문/입력 시점·worker readout·퇴역
/// 권한이 없으므로 실제 수집 row의 입력이나 WDL target에 붙일 수 없다.
pub struct RustNnuecReference {
    anchor: GenerationRulesAnchorV3,
    value: Value,
    model_sha256: String,
    semantics: String,
}
impl RustNnuecReference {
    pub fn anchor(&self) -> &GenerationRulesAnchorV3 {
        &self.anchor
    }
    pub fn reference_input(&self) -> &Value {
        &self.value
    }
    pub fn model_sha256(&self) -> &str {
        &self.model_sha256
    }
    pub fn semantics(&self) -> &str {
        &self.semantics
    }
    pub fn v1(
        model: &rz_search::cpu_value::nnuec::NnuecModel,
        context: &rz_search::cpu_value::nnuec::NnuecContext,
        local: &rz_search::cpu_value::nnuec::NnuecLocal,
        position: &Position,
    ) -> Result<Self, ArenaError> {
        let input = model
            .reference_input(context, local, position)
            .map_err(|e| invalid(e.to_string()))?;
        Ok(Self {
            anchor: exact_rules_anchor(position)?,
            value: serde_json::to_value(input).map_err(|e| invalid(e.to_string()))?,
            model_sha256: model
                .identity()
                .weights_sha256
                .clone()
                .filter(|v| generation_sha(v))
                .ok_or_else(|| invalid("NNUEC model weights not identified"))?,
            semantics: model.identity().semantics.clone(),
        })
    }
    pub fn v2(
        model: &rz_search::cpu_value::nnuec::v2::Model,
        context: &rz_search::cpu_value::nnuec::v2::Context,
        local: &rz_search::cpu_value::nnuec::NnuecLocal,
        position: &Position,
    ) -> Result<Self, ArenaError> {
        let input = model
            .reference_input(context, local, position)
            .map_err(|e| invalid(e.to_string()))?;
        Ok(Self {
            anchor: exact_rules_anchor(position)?,
            value: serde_json::to_value(input).map_err(|e| invalid(e.to_string()))?,
            model_sha256: model
                .identity()
                .weights_sha256
                .clone()
                .filter(|v| generation_sha(v))
                .ok_or_else(|| invalid("NNUEC V2 model weights not identified"))?,
            semantics: model.identity().semantics.clone(),
        })
    }
}

/// A bounded JSONL packet. The caller retains filesystem ownership, atomic
/// publication, raw-data retention and the shared staging quota.
pub struct RoventChunkPacket {
    pub declaration: RoventCorpusChunk,
    pub bytes: Vec<u8>,
}
pub struct RoventCorpusWriter {
    manifest: RoventCorpusManifest,
    index: RoventCorpusIndex,
    buffer: Vec<u8>,
    buffer_rows: u64,
    total_bytes: u64,
    record_seals: BTreeSet<String>,
    // Only metadata snapshots survive publication; old root Rules/model pins
    // are not kept alive by the corpus writer.
    admissions: BTreeMap<String, std::sync::Arc<publication::RoventPublicationSourceSnapshot>>,
    publication_sources:
        BTreeMap<String, std::sync::Arc<publication::RoventPublicationSourceSnapshot>>,
    publication_payloads:
        BTreeMap<String, std::sync::Arc<publication::RoventPublicationSourceSnapshot>>,
    allocations: allocation::RoventAllocationSupplementCapture,
}
impl RoventCorpusWriter {
    pub fn new(manifest: &RoventCorpusManifest) -> Result<Self, ArenaError> {
        let manifest_bytes = serde_json::to_vec(manifest)
            .map_err(|e| invalid(e.to_string()))?
            .len() as u64;
        Ok(Self {
            manifest: manifest.clone(),
            index: RoventCorpusIndex {
                schema: INDEX_DOMAIN.into(),
                manifest_sha256: manifest.sha256()?,
                record_count: 0,
                chunks: Vec::new(),
                origins: Vec::new(),
            },
            buffer: Vec::new(),
            buffer_rows: 0,
            // Conservative reserve for changing index counters/framing.
            total_bytes: manifest_bytes + 4096,
            record_seals: BTreeSet::new(),
            admissions: BTreeMap::new(),
            publication_sources: BTreeMap::new(),
            publication_payloads: BTreeMap::new(),
            allocations: allocation::RoventAllocationSupplementCapture::new(),
        })
    }
    pub fn append(&mut self, row: PublishedRoventRecord) -> Result<(), ArenaError> {
        check(
            row.manifest_sha256 == self.index.manifest_sha256,
            "published row belongs to another corpus writer",
        )?;
        row.record.validate_rules_replay_v2(&self.manifest)?;
        check(
            row.record.native.raw_retirement["root_admission_sha256"] == row.admission.raw_sha256()
                && row.record.native.raw_retirement["root_admission"]
                    == *row.admission.raw_receipt(),
            "writer requires the original published root admission",
        )?;
        let mut bytes = serde_json::to_vec(&row.record).map_err(|e| invalid(e.to_string()))?;
        bytes.push(b'\n');
        self.append_original_parts(
            &row.admission,
            &row.record.record_sha256,
            RoventCorpusOrigin::from_record(&row.record),
            bytes,
        )
    }
    /// 호출 형식의 전체 검증 뒤 원 admission과 정확 serialized bytes를 이동한다.
    /// metadata decoder가 이 private 경로를 호출하여 PublishedRecord를 발급할 수 없다.
    fn append_original_parts(
        &mut self,
        admission: &RoventCollectionRootAdmission,
        record_sha256: &str,
        origin: RoventCorpusOrigin,
        bytes: Vec<u8>,
    ) -> Result<(), ArenaError> {
        let admission_sha = admission.raw_sha256().to_owned();
        let source_sha = admission.native_origin().raw_sha256().to_owned();
        // Read/serialize original source metadata once per native origin, then
        // share equal source payloads across process/root occurrences.
        let (source, new_payload) = match self.publication_sources.get(&source_sha) {
            Some(source) => (std::sync::Arc::clone(source), None),
            None => {
                let snapshot = publication::RoventPublicationSourceSnapshot::capture(admission)?;
                let key = snapshot.payload_key();
                match self.publication_payloads.get(&key) {
                    Some(original) => {
                        check(
                            original.same_source(&snapshot),
                            "conflicting original publication source",
                        )?;
                        (std::sync::Arc::clone(original), None)
                    }
                    None => (std::sync::Arc::new(snapshot), Some(key)),
                }
            }
        };
        if let Some(original) = self.admissions.get(&admission_sha) {
            check(
                std::sync::Arc::ptr_eq(original, &source),
                "root admission changed its original source",
            )?;
        }
        let origin_bytes = serde_json::to_vec(&origin)
            .map_err(|e| invalid(e.to_string()))?
            .len() as u64
            + 1;
        let source_bytes = if new_payload.is_some() {
            source.payload_bytes()?
        } else {
            0
        };
        let admission_bytes = if self.admissions.contains_key(&admission_sha) {
            0
        } else {
            1024 + admission
                .root_admission()
                .allocation_clock_canonical_bytes()
                .len() as u64
        };
        let added = (bytes.len() as u64)
            .checked_add(origin_bytes)
            .and_then(|n| n.checked_add(source_bytes))
            .and_then(|n| n.checked_add(admission_bytes))
            .ok_or_else(|| invalid("corpus byte count overflow"))?;
        check(
            self.index.origins.len() < MAX_CORPUS_ROWS
                && bytes.len() <= GENERATION_MAX_ROW_BYTES + 1
                && self
                    .buffer
                    .len()
                    .checked_add(bytes.len())
                    .is_some_and(|n| n <= MAX_CHUNK_BYTES)
                && self
                    .total_bytes
                    .checked_add(added)
                    .is_some_and(|n| n <= 18 * 1024 * 1024 * 1024)
                && !self.record_seals.contains(record_sha256),
            "corpus output cap/duplicate; flush full chunk explicitly",
        )?;
        self.allocations.retain_admission(admission)?;
        self.record_seals.insert(record_sha256.to_owned());
        if let Some(key) = new_payload {
            self.publication_payloads
                .insert(key, std::sync::Arc::clone(&source));
        }
        self.publication_sources
            .entry(source_sha)
            .or_insert_with(|| std::sync::Arc::clone(&source));
        self.admissions.entry(admission_sha).or_insert(source);
        self.index.origins.push(origin);
        self.index.record_count += 1;
        self.buffer_rows += 1;
        self.total_bytes += added;
        self.buffer.extend(bytes);
        Ok(())
    }
    pub fn flush(&mut self) -> Result<Option<RoventChunkPacket>, ArenaError> {
        if self.buffer.is_empty() {
            return Ok(None);
        }
        check(self.index.chunks.len() < 8192, "corpus chunk count cap")?;
        let declaration_bytes_reserve = 512u64;
        check(
            self.total_bytes
                .checked_add(declaration_bytes_reserve)
                .is_some_and(|n| n <= 18 * 1024 * 1024 * 1024),
            "corpus chunk/index publication cap",
        )?;
        self.total_bytes += declaration_bytes_reserve;
        let bytes = std::mem::take(&mut self.buffer);
        let declaration = RoventCorpusChunk {
            path: format!("chunks/{:06}.jsonl", self.index.chunks.len()),
            bytes: bytes.len() as u64,
            sha256: hex(Digest(Sha256::digest(&bytes).into())),
            records: self.buffer_rows,
        };
        self.buffer_rows = 0;
        self.index.chunks.push(declaration.clone());
        Ok(Some(RoventChunkPacket { declaration, bytes }))
    }
    pub fn finish(self) -> Result<RoventCorpusIndex, ArenaError> {
        check(
            self.buffer.is_empty(),
            "flush original chunk before corpus index publication",
        )?;
        self.index.validate()?;
        Ok(self.index)
    }
    /// Consumes only this writer's original opaque published records. The
    /// returned owner can commit durable data, never restore native authority.
    pub fn finish_publication(self) -> Result<FinishedRoventCorpus, ArenaError> {
        FinishedRoventCorpus::from_writer(self)
    }
}

#[cfg(feature = "pals-collection-onnx")]
mod teacher;
#[cfg(feature = "pals-collection-onnx")]
pub use teacher::{
    RoventCollectionTeacherBinding, RoventCollectionTeacherCompleted,
    RoventCollectionTeacherFailure, RoventCollectionTeacherPump, RoventCollectionTeacherSession,
    RoventCollectionTeacherShutdown, RoventRegisteredTeacher, RoventTeacherBinding,
};
#[cfg(feature = "pals-collection-onnx")]
mod native_capture;
#[cfg(feature = "pals-collection-onnx")]
pub use native_capture::*;
#[cfg(all(target_os = "linux", feature = "pals-collection-onnx"))]
mod collection_hooks_core_v2;
#[cfg(feature = "pals-collection-onnx")]
mod observer;
#[cfg(feature = "pals-collection-onnx")]
mod observer_lock_v2;
#[cfg(all(target_os = "linux", feature = "pals-collection-onnx"))]
#[allow(dead_code, unused_imports)]
mod own_hooks_lock_v2;
#[cfg(all(target_os = "linux", feature = "pals-collection-onnx"))]
mod pc_hooks_lock_v2;
#[cfg(feature = "pals-collection-onnx")]
pub use observer::*;
#[cfg(feature = "pals-collection-onnx")]
pub use observer_lock_v2::*;
#[cfg(all(target_os = "linux", feature = "pals-collection-onnx"))]
#[allow(unused_imports)]
pub(super) use pc_hooks_lock_v2::{
    RoventPcCollectionBridgeV2, RoventPcCollectionWorkerExitV2, RoventPcCollectionWorkerV2,
};
mod rules_replay;
pub use rules_replay::*;
mod external_scope;
mod historical_declarations_v2;
mod historical_native_lock_v2;
pub use historical_native_lock_v2::{
    VerifiedHistoricalNativeRootMetadataV2, verify_historical_lock_v2_record_evidence,
    verify_historical_native_root_metadata_v2,
};
mod historical_scope;
pub use external_scope::{
    RoventHistoricalExternalSource, verify_historical_external_source_input_lock,
};
pub use historical_scope::{RoventHistoricalRecordEvidence, verify_historical_record_evidence};
#[cfg(target_os = "linux")]
pub(crate) use historical_scope::{RoventNativeOriginEvidence, RoventRootAdmissionEvidence};
mod publication;
pub use publication::{
    FinishedRoventCorpus, ROVENT_CORPUS_PUBLICATION, ROVENT_PUBLICATION_FILE,
    ROVENT_PUBLICATION_INDEX, ROVENT_PUBLICATION_MANIFEST, RoventCorpusPublication,
    RoventPublicationAdmission, RoventPublicationFile,
};
mod publication_v2;
pub use publication_v2::{ROVENT_CORPUS_PUBLICATION_V2, RoventCorpusPublicationV2};
#[cfg(all(target_os = "linux", feature = "pals-collection-onnx"))]
mod publication_lock_v2;
mod submitted_lock_v2;
mod submitted_v3;
pub use submitted_lock_v2::{PublishedRoventLockV2Record, RoventRetiredLockV2Record};
pub use submitted_v3::{
    PublishedRoventRecordV2, RoventRetiredRecordV2, RoventSubmittedRecordMetadata,
    RoventSubmittedReviewEvidence, capture_rovent_submitted_input,
};
mod producer_v2;
pub use producer_v2::{
    RoventCapturedInputV2, RoventCorpusProducerLockV2, RoventCorpusProducerV2,
    RoventInputCaptureReservationV2,
};
mod writer_v2;
pub use writer_v2::{FinishedRoventCorpusV2, RoventCorpusWriterV2};
#[cfg(target_os = "linux")]
mod writer_lock_v2;
#[cfg(target_os = "linux")]
pub use writer_lock_v2::{FinishedRoventCorpusLockV2, RoventCorpusWriterLockV2};
#[cfg(all(target_os = "linux", feature = "pals-collection-onnx"))]
mod publication_caller_lock_v2;
#[cfg(all(target_os = "linux", feature = "pals-collection-onnx"))]
pub use publication_caller_lock_v2::RoventCorpusAssemblyLockV2;
mod historical_v2;
mod submitted_review_historical;
pub use historical_v2::verify_historical_submitted_record_evidence;
mod divergence_owner;
pub use divergence_owner::verify_historical_pc_divergence;
mod r_readout;
pub use r_readout::*;
#[cfg(feature = "pals-collection-onnx")]
mod r_observer;
#[cfg(feature = "pals-collection-onnx")]
pub use r_observer::*;
#[cfg(feature = "pals-collection-onnx")]
mod r_semantic;
#[cfg(feature = "pals-collection-onnx")]
pub use r_semantic::*;
#[cfg(feature = "pals-collection-onnx")]
mod r_unsubmitted;
#[cfg(feature = "pals-collection-onnx")]
pub use r_unsubmitted::*;
#[cfg(feature = "pals-collection-onnx")]
mod r_semantic_materialized;
#[cfg(feature = "pals-collection-onnx")]
pub use r_semantic_materialized::*;
#[cfg(feature = "pals-collection-onnx")]
mod r_cohort_materialized;
#[cfg(feature = "pals-collection-onnx")]
pub use r_cohort_materialized::*;
#[cfg(feature = "pals-collection-onnx")]
mod utility_cohort_owner;
#[cfg(feature = "pals-collection-onnx")]
pub use utility_cohort_owner::*;
#[cfg(feature = "pals-collection-onnx")]
mod utility_intervention_owner;
#[cfg(feature = "pals-collection-onnx")]
pub use utility_intervention_owner::*;
#[cfg(feature = "pals-collection-onnx")]
mod utility_owner;
#[cfg(feature = "pals-collection-onnx")]
pub use utility_owner::*;
mod r_publication;
pub use r_publication::*;
mod r_writer;
pub use r_writer::*;
mod r_historical;
pub use r_historical::*;
mod allocation;
#[cfg(target_os = "linux")]
pub mod launch;
pub use allocation::{
    ROVENT_ALLOCATION_SUPPLEMENT, ROVENT_R_ALLOCATION_SUPPLEMENT, RoventAllocationSupplement,
    RoventAllocationSupplementEntry,
};
#[cfg(all(test, target_os = "linux"))]
mod r_wire_tests;

#[cfg(test)]
mod tests {
    use super::*;
    fn metadata_producer() -> RoventCorpusProducer {
        RoventCorpusProducer::new(RoventCorpusManifest {
            schema: CORPUS_DOMAIN.into(),
            corpus_id: "metadata-only".into(),
            source_commit: "a".repeat(40),
            source_sha256: "a".repeat(64),
            runtime_sha256: "b".repeat(64),
            model_semantics_sha256: "c".repeat(64),
            model_epoch: 1,
            context_epoch: 1,
            recipe_sha256: "d".repeat(64),
            stage_id: "stage".into(),
            round: 0,
            round_id: "common".into(),
            student: "student".into(),
            learner: "student".into(),
            one_learner: true,
            inherited_input_sha256: vec![],
            inherited_origin_sha256: vec![],
            teachers: BTreeMap::new(),
            opening_splits: BTreeMap::from([("opening".into(), GenerationSplitV3::Train)]),
            ancestor_index_sha256: vec![],
        })
        .unwrap()
    }
    fn metadata_context() -> GenerationCaptureContextV3 {
        use rz_contracts::pals::{RepresentationKey, SearchAuthority};
        use rz_contracts::pals_adaptive::ReviewViewSeal;
        use rz_contracts::{
            ClockDomain, Deadline, GameGeneration, MonotonicTick, OwnerId, PrecisionProfile,
            ProcessEpoch, RootGeneration,
        };
        let position = Position::startpos();
        let state = rz_position::contracts::ContractPosition::new(OwnerId(1), position.clone())
            .export()
            .unwrap();
        let authority = SearchAuthority {
            epoch: ProcessEpoch(1),
            game: GameGeneration(1),
            root: RootGeneration(1),
            implementation: Digest([1; 32]),
        };
        let ticket = RoleTicketEnvelope {
            id: RequestId::new(authority.epoch, 1),
            authority,
            question: Digest([2; 32]),
            role: Role::Proposer,
            view: ReviewViewSeal {
                authority,
                view_sequence: 1,
                context_policy: Digest([3; 32]),
                view_digest: Digest([4; 32]),
                root_state: state.snapshot().identity(),
                actual_tokens: 0,
                selected_records: 0,
                anchors: 1,
                length_bucket: 8,
            },
            representation: RepresentationKey {
                input: Digest([5; 32]),
                history: Digest([6; 32]),
                candidates: Digest([7; 32]),
                mask_positions: Digest([8; 32]),
                model: Digest([9; 32]),
                encoding: Digest([10; 32]),
                precision: PrecisionProfile::Fp32,
                frozen_epoch: 1,
                record_revision: 0,
                role: Role::Proposer,
            },
            original_deadline: Deadline {
                clock: ClockDomain(ProcessEpoch(1)),
                at: MonotonicTick(100),
            },
        };
        GenerationCaptureContextV3 {
            game_id: "game".into(),
            opening_id: "opening".into(),
            line_genealogy_id: "line".into(),
            model_epoch: 1,
            context_epoch: 1,
            capture_sequence: 0,
            work_scope: crate::pals_collect_v3::GenerationCaptureWorkScopeV3::from_native_ticket(
                &ticket,
            ),
            actual_position: position,
            actual_history: vec![],
            action_support: crate::pals_collect_v3::GenerationActionSupportV3::non_validator(),
        }
    }
    #[test]
    fn bounded_capture_reservations_cancel_without_native_authority() {
        // 예약 메타데이터 검사다. native worker·model·accepted step은 없다.
        let mut p = metadata_producer();
        let mut reservations = Vec::new();
        for _ in 0..16 {
            reservations.push(
                p.reserve_input(
                    metadata_context(),
                    RoventLineage {
                        actor_id: "student".into(),
                        teacher_actor_id: None,
                        ancestor_origin_sha256: vec![],
                    },
                )
                .unwrap(),
            );
        }
        assert_eq!(p.incomplete_capture_count(), 16);
        assert!(
            p.reserve_input(
                metadata_context(),
                RoventLineage {
                    actor_id: "student".into(),
                    teacher_actor_id: None,
                    ancestor_origin_sha256: vec![]
                }
            )
            .is_err()
        );
        let sequence = reservations.remove(0).sequence();
        assert!(p.discard_reserved_input(sequence));
        assert!(!p.discard_reserved_input(sequence));
        assert_eq!(p.incomplete_capture_count(), 15);
        assert!(
            p.submitted_input_seal(RequestId::new(rz_contracts::ProcessEpoch(1), 1))
                .is_err()
        );
    }
    #[test]
    fn rovent_rules_owner_refuses_unfinished_and_keeps_actual_trajectory() {
        let mut game =
            RoventGameTrajectory::begin("rovent-mate".into(), Position::startpos()).unwrap();
        assert!(game.finish_rules().is_err());
        let original = exact_rules_anchor(game.position()).unwrap();
        for m in ["f2f3", "e7e5", "g2g4", "d8h4"] {
            game.play_selected_move(BoardMove::from_uci(m).unwrap())
                .unwrap();
        }
        let outcome = game.finish_rules().unwrap();
        assert_eq!(outcome.authority.ending, GenerationRulesEndingV3::Checkmate);
        assert_eq!(outcome.authority.outcome, GenerationOutcomeV3::BlackWin);
        assert!(outcome.authority.anchors.contains(&original));
        assert_eq!(outcome.authority.anchors.len(), 5);
        assert!(game.finish_rules().is_err());
    }
    #[test]
    fn rovent_rules_anchor_does_not_merge_imported_fen_with_played_history() {
        let mut played = Position::startpos();
        for m in ["g1f3", "g8f6", "f3g1", "f6g8"] {
            played.make_move(BoardMove::from_uci(m).unwrap()).unwrap();
        }
        let imported = Position::from_fen(&played.to_fen()).unwrap();
        let a = exact_rules_anchor(&played).unwrap();
        let b = exact_rules_anchor(&imported).unwrap();
        assert_eq!(a.board_fen, b.board_fen);
        assert_ne!(a.rules_history_sha256, b.rules_history_sha256);
    }
}
