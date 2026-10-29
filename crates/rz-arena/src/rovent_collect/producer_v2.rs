//! submitted-input/1 전용 원 native 수집 소유자다. 원 ticket/완료/수락 처리는
//! 기존 유한 owner 경로를 공유하고 형식별 입력과 raw/3 봉인은 별도로 유지한다.
use super::*;
use rz_experiments::rovent_corpus_v2::{
    RoventCorpusManifestV2, RoventTargetsV2, RoventTrainingRecordV2,
};
use rz_experiments::rovent_submitted_input::{
    ROVENT_NATIVE_RAW_V3_DOMAIN, ROVENT_TRAINING_RECORD_V2_DOMAIN,
};

pub struct RoventInputCaptureReservationV2 {
    original: RoventInputCaptureReservation,
    native_domain: &'static str,
}
pub struct RoventCapturedInputV2 {
    original: RoventCapturedInput,
    input: rz_experiments::rovent_submitted_input::RoventSubmittedInputV1,
}
impl RoventInputCaptureReservationV2 {
    pub fn sequence(&self) -> u64 {
        self.original.sequence()
    }
    pub fn capture(
        self,
        ticket: RoleTicketEnvelope,
        view: &ReviewView,
        input: &PalsVirtualReviewInput,
        identity: &ReviewExecutionIdentity,
    ) -> Result<RoventCapturedInputV2, ArenaError> {
        input
            .validate()
            .map_err(|error| invalid(error.to_string()))?;
        check(
            input
                .fresh_key(identity)
                .map_err(|error| invalid(error.to_string()))?
                .0
                == ticket.representation.input.0,
            "raw3 captured input differs from original native representation",
        )?;
        let (captured, review) =
            capture_rovent_submitted_input(&ticket, view, input, &self.original.context)?;
        let input_sha256 = captured.sha256()?;
        let replay =
            RoventSubmittedRulesReplay::capture(&self.original.context.actual_position, view)?;
        let raw = json!({"schema":self.native_domain,"kind":"submitted","sequence":self.original.context.capture_sequence,
            "process_epoch":ticket.id.epoch.0,"game_generation":ticket.authority.game.0,"root_generation":ticket.authority.root.0,
            "request_sequence":ticket.id.sequence,"head":role_name(ticket.role),"source_sha256":self.original.source_sha256,
            "model_sha256":hex(ticket.representation.model),"encoding_sha256":hex(ticket.representation.encoding),
            "input_digest":hex(ticket.representation.input),"view_digest":hex(ticket.view.view_digest),
            "input_sha256":input_sha256,"ticket":ticket_json(&ticket),"rules_replay":replay,"submitted_review":review});
        check(
            serde_json::to_vec(&raw)
                .map_err(|error| invalid(error.to_string()))?
                .len()
                <= GENERATION_MAX_ROW_BYTES,
            "raw3 original submission exceeds bounded row cap before native dispatch",
        )?;
        // 이 내부 projection은 물리 owner 처리의 공통 scalar만 전달한다.
        // legacy input/record로 검증·직렬화·게시하는 경로에 전달하지 않는다.
        let common = captured.common_validation_projection();
        Ok(RoventCapturedInputV2 {
            input: captured,
            original: RoventCapturedInput {
                manifest_sha256: self.original.manifest_sha256,
                pending: Pending {
                    ticket,
                    input: common,
                    input_sha256,
                    lineage: self.original.lineage,
                    submitted: self.original.context.capture_sequence,
                    raw_submission: raw,
                    completed: None,
                },
            },
        })
    }
}
impl RoventCapturedInputV2 {
    pub fn submitted_input_seal(&self) -> RoventSubmittedInputSeal {
        self.original.submitted_input_seal()
    }
}

pub struct RoventCorpusProducerV2 {
    manifest: RoventCorpusManifestV2,
    original: RoventCorpusProducer,
    inputs: BTreeMap<(u64, u64), rz_experiments::rovent_submitted_input::RoventSubmittedInputV1>,
}

/// Preparation and native-event custody for the distinct input-lock /2
/// corpus family. This is not a source of retirement or publication authority:
/// an accepted manager-issued witness must still be joined to the original
/// physical completion before an opaque row can be emitted.
pub struct RoventCorpusProducerLockV2 {
    manifest: rz_experiments::rovent_corpus_lock_v2::record::RoventCorpusManifestLockV2,
    admission: rz_uci::pals_adaptive::RoventCollectionRootAdmissionV2,
    original: RoventCorpusProducerV2,
    native_retirements: BTreeMap<RequestId, RoleTicketEnvelope>,
    native_observation_failures: BTreeMap<RequestId, Vec<rz_uci::pals_native::ticketed::TicketNativeObservationFailure>>,
}

impl RoventCorpusProducerLockV2 {
    pub fn new(
        manifest: rz_experiments::rovent_corpus_lock_v2::record::RoventCorpusManifestLockV2,
        admission: rz_uci::pals_adaptive::RoventCollectionRootAdmissionV2,
    ) -> Result<Self, ArenaError> {
        let lock = admission.native_origin().original_input_lock();
        manifest.validate(lock)?;
        rz_experiments::rovent_corpus_lock_v2::native::validate_lock_v2_root_metadata(
            admission.raw_receipt(),
            &json!(admission.raw_sha256()),
            lock,
        )?;
        let manifest_sha256 = manifest.sha256(lock)?;
        let mut original = RoventCorpusProducerV2::new(manifest.registration.clone())?;
        original.original.manifest_sha256 = manifest_sha256;
        original.original.native_domain =
            rz_experiments::rovent_corpus_lock_v2::native::ROVENT_LOCK_V2_NATIVE_RAW_DOMAIN;
        Ok(Self {
            manifest,
            admission,
            original,
            native_retirements: BTreeMap::new(),
            native_observation_failures: BTreeMap::new(),
        })
    }

    pub fn manifest(&self) -> &rz_experiments::rovent_corpus_lock_v2::record::RoventCorpusManifestLockV2 {
        &self.manifest
    }

    pub fn incomplete_capture_count(&self) -> usize {
        self.original.incomplete_capture_count()
    }

    /// Exact owner observations retained with the original ticket. These are
    /// collection failure evidence and never replace a physical terminal.
    pub fn native_observation_failures(
        &self,
        id: RequestId,
    ) -> Option<&[rz_uci::pals_native::ticketed::TicketNativeObservationFailure]> {
        self.native_observation_failures.get(&id).map(Vec::as_slice)
    }

    pub(super) fn validate_admission(
        &self,
        admission: &rz_uci::pals_adaptive::RoventCollectionRootAdmissionV2,
    ) -> Result<(), ArenaError> {
        check(self.admission.same_original_admission(admission),
            "lock-v2 collection admission differs from original owner")
    }

    /// Reserve only for a ticket whose manager dispatch witness was checked by
    /// the collection observer. The native callback owns the actual capture.
    pub(super) fn reserve_input(
        &mut self,
        ticket: &RoleTicketEnvelope,
        context: GenerationCaptureContextV3,
        lineage: RoventLineage,
    ) -> Result<RoventInputCaptureReservationV2, ArenaError> {
        self.admission
            .validate_ticket(ticket)
            .map_err(|error| invalid(error.to_string()))?;
        self.original.reserve_input(context, lineage)
    }

    pub(super) fn admit_captured_lock_input(
        &mut self,
        input: super::native_capture::RoventCapturedLockInputV2,
    ) -> Result<(), Box<(ArenaError, super::native_capture::RoventCapturedLockInputV2)>> {
        // The private receiver only admits the registered native callback.
        // Do not require a still-live root here: a late completion must remain
        // drainable after logical cancellation, without regaining authority.
        let validation = (|| {
            self.validate_admission(input.admission())?;
            let captured = input.captured();
            let key = (captured.original.pending.ticket.id.epoch.0,
                captured.original.pending.ticket.id.sequence);
            let original = &self.original.original;
            check(!self.original.inputs.contains_key(&key) && self.original.inputs.len() < 16,
                "lock-v2 original input duplicate/cap")?;
            check(captured.original.manifest_sha256 == original.manifest_sha256
                && original.reserved.contains(&captured.original.pending.submitted)
                && original.pending.len() < 16 && !original.pending.contains_key(&key)
                && !original.retired.contains(&key),
                "lock-v2 revoked/different-owner/duplicate captured input")?;
            Ok::<(), ArenaError>(())
        })();
        if let Err(error) = validation {
            return Err(Box::new((error, input)));
        }
        // Validation happens before moving the input; commit has no fallible
        // operations and preserves the original owner and reservation.
        let (captured, _) = input.into_original();
        let key = (captured.original.pending.ticket.id.epoch.0,
            captured.original.pending.ticket.id.sequence);
        self.original.original.reserved.remove(&captured.original.pending.submitted);
        self.original.original.pending.insert(key, captured.original.pending);
        self.original.inputs.insert(key, captured.input);
        Ok(())
    }

    pub(super) fn discard_reserved_input(&mut self, sequence: u64) -> bool {
        self.original.discard_reserved_input(sequence)
    }

    /// Only a manager-issued original-worker event can reach the physical
    /// completion accounting. The witness does not itself establish completion.
    pub fn observe_native_event<O: ProposerRankingOutput>(
        &mut self,
        witness: rz_uci::pals_adaptive::RoventCollectionNativeEventWitnessV2<'_, O>,
    ) -> Result<(), ArenaError> {
        check(
            self.admission.same_original_admission(witness.admission()),
            "lock-v2 native event belongs to another original admission",
        )?;
        let event = witness.event();
        let (ticket, failures) = match event {
            TicketNativeEvent::Completed { completion, observation_failures, .. } =>
                (Some(completion.ticket), Some(observation_failures)),
            TicketNativeEvent::Retired { ticket, observation_failures, .. } =>
                (Some(*ticket), Some(observation_failures)),
            _ => (None, None),
        };
        if let (Some(ticket), Some(failures)) = (ticket, failures) {
            if !failures.is_empty() {
                check(self.native_observation_failures.len() < MAX_CORPUS_ROWS
                    || self.native_observation_failures.contains_key(&ticket.id),
                    "lock-v2 native failure evidence cap")?;
                let records = self.native_observation_failures.entry(ticket.id).or_default();
                records.extend(failures.iter().cloned());
                return Err(invalid(format!("lock-v2 original native observation failed: {records:?}")));
            }
        }
        self.original.observe_native_event(event)?;
        if let TicketNativeEvent::Retired { ticket, accepted: true, result: Ok(()), .. } = event {
            check(!self.native_observation_failures.contains_key(&ticket.id)
                && self.native_retirements.len() < MAX_CORPUS_ROWS
                && !self.native_retirements.contains_key(&ticket.id),
                "lock-v2 duplicate or failed native retirement")?;
            self.native_retirements.insert(ticket.id, *ticket);
        }
        Ok(())
    }

    pub fn discard_unpublished(&mut self, id: RequestId) -> bool {
        self.original.discard_unpublished(id)
    }

    /// Consumes the manager-issued witness for an actually accepted and
    /// search-consumed result. The pending service completion is checked
    /// separately; the witness itself is not a physical-completion claim.
    pub fn retire_consumed(
        &mut self,
        witness: rz_uci::pals_adaptive::RoventCollectionConsumedWitnessV2<'_>,
    ) -> Result<RoventRetiredLockV2Record, ArenaError> {
        check(
            self.admission.same_original_admission(witness.admission()),
            "lock-v2 consumption belongs to another original admission",
        )?;
        let accepted = witness.accepted();
        self.retire_original(*accepted.ticket(), accepted.output_digest(), "accepted_role_step")
    }

    /// The manager's original root-ranking publication is a separate proof;
    /// it must match this owner and the actual native output and completion.
    pub fn retire_root_ranking(
        &mut self,
        accepted: &rz_uci::pals_adaptive::RoventRootRankingAcceptanceV2,
        output: &RoleOutput,
    ) -> Result<RoventRetiredLockV2Record, ArenaError> {
        check(
            self.admission.same_original_admission(accepted.admission()),
            "lock-v2 ranking belongs to another original admission",
        )?;
        let source = accepted.ranking().source();
        accepted.admission().validate_output(&source.ticket, output)
            .map_err(|error| invalid(error.to_string()))?;
        check(
            native_acceptance_output_digest(output).map_err(|error| invalid(error.to_string()))?
                == source.output_digest,
            "lock-v2 original ranking/output digest mismatch",
        )?;
        self.retire_original(source.ticket, source.output_digest, "root_ranking_published")
    }

    fn retire_original(
        &mut self,
        ticket: RoleTicketEnvelope,
        output_digest: Digest,
        acceptance_kind: &str,
    ) -> Result<RoventRetiredLockV2Record, ArenaError> {
        check(self.native_retirements.get(&ticket.id) == Some(&ticket)
            && !self.native_observation_failures.contains_key(&ticket.id),
            "lock-v2 original native retirement is missing or reported an observation failure")?;
        let key = (ticket.id.epoch.0, ticket.id.sequence);
        let producer = &mut self.original.original;
        let pending = producer.pending.get(&key)
            .ok_or_else(|| invalid("lock-v2 retirement has no original input"))?;
        let completed = pending.completed.as_ref()
            .ok_or_else(|| invalid("lock-v2 retirement before physical completion"))?;
        let input = self.original.inputs.get(&key)
            .ok_or_else(|| invalid("lock-v2 retained input is missing"))?;
        check(
            pending.ticket == ticket
                && completed.output_digest == hex(output_digest)
                && input.sha256()? == pending.input_sha256
                && !producer.retired.contains(&key)
                && producer.retired.len() < MAX_CORPUS_ROWS,
            "lock-v2 accepted result differs from original physical completion",
        )?;
        let split = *producer.manifest.opening_splits.get(&input.opening_id)
            .ok_or_else(|| invalid("lock-v2 opening split is not registered"))?;
        let sequence = producer.sequence.checked_add(1)
            .ok_or_else(|| invalid("Rovent collector sequence exhausted"))?;
        let raw = json!({"schema":producer.native_domain,"kind":"retired","sequence":sequence,
            "process_epoch":ticket.id.epoch.0,"game_generation":ticket.authority.game.0,"root_generation":ticket.authority.root.0,
            "request_sequence":ticket.id.sequence,"head":role_name(ticket.role),"source_sha256":producer.manifest.source_sha256,
            "input_sha256":pending.input_sha256,"submitted_sha256":completed.raw_submission_sha256,
            "completed_sha256":completed.raw_completion_sha256,"output_digest":completed.output_digest,
            "accepted_retired":true,"ticket":ticket_json(&ticket),"acceptance_kind":acceptance_kind,
            "root_admission":self.admission.raw_receipt(),"root_admission_sha256":self.admission.raw_sha256()});
        let raw_sha256 = generation_digest(producer.native_domain, &raw)?;
        // No fallible operations remain before the original pending/input
        // owners are transferred to the retired row.
        producer.sequence = sequence;
        let pending = producer.pending.remove(&key).expect("checked pending");
        let input = self.original.inputs.remove(&key).expect("checked input");
        let mut native = pending.completed.expect("checked physical completion");
        native.raw_retirement = raw;
        native.raw_retirement_sha256 = raw_sha256;
        native.retired_sequence = sequence;
        native.accepted_retired = true;
        producer.retired.insert(key);
        Ok(RoventRetiredLockV2Record {
            manifest_sha256: producer.manifest_sha256.clone(),
            ticket,
            admission: self.admission.clone(),
            record: rz_experiments::rovent_corpus_lock_v2::record::RoventTrainingRecordLockV2(
                RoventTrainingRecordV2 {
                    schema: rz_experiments::rovent_corpus_lock_v2::record::ROVENT_TRAINING_RECORD_LOCK_V2_DOMAIN.into(),
                    role: pending.input.role,
                    split,
                    input,
                    input_sha256: pending.input_sha256,
                    origin_sha256: String::new(),
                    record_sha256: String::new(),
                    lineage: pending.lineage,
                    native,
                    head_sources: BTreeMap::new(),
                    targets: RoventTargetsV2::default(),
                    divergence_index: None,
                    nnuec_input: None,
                    nnuec_input_sha256: None,
                    nnuec_model_sha256: None,
                    nnuec_semantics: None,
                    nnuec_wdl: None,
                },
            ),
        })
    }

    /// Keep the original row with the caller on validation failure.
    pub fn seal_retired(
        &self,
        row: &mut RoventRetiredLockV2Record,
    ) -> Result<(), ArenaError> {
        check(
            row.manifest_sha256 == self.original.original.manifest_sha256
                && row.admission.same_original_admission(&self.admission)
                && row.record.0.native.raw_retirement["root_admission_sha256"]
                    == self.admission.raw_sha256()
                && row.record.0.native.raw_retirement["root_admission"]
                    == *self.admission.raw_receipt(),
            "lock-v2 retired record lost original owner",
        )?;
        let mut candidate = row.record.clone();
        candidate.seal(&self.manifest, self.admission.native_origin().original_input_lock())?;
        row.record = candidate;
        Ok(())
    }

    /// Publication of the in-process row does not write files. On failure the
    /// caller gets the original row and its admission back for recovery.
    pub fn publish(
        &self,
        mut row: RoventRetiredLockV2Record,
    ) -> Result<PublishedRoventLockV2Record, Box<(ArenaError, RoventRetiredLockV2Record)>> {
        if let Err(error) = self.seal_retired(&mut row) {
            return Err(Box::new((error, row)));
        }
        Ok(PublishedRoventLockV2Record {
            record: row.record,
            manifest_sha256: row.manifest_sha256,
            admission: row.admission,
        })
    }

    /// Attach only an actual completed Rules trajectory containing this
    /// exact input anchor. No teacher score is converted into a WDL target.
    pub fn attach_actual_wdl(
        &mut self,
        row: &mut RoventRetiredLockV2Record,
        outcome: &RoventRulesOutcome,
    ) -> Result<(), ArenaError> {
        check(
            row.manifest_sha256 == self.original.original.manifest_sha256
                && row.admission.same_original_admission(&self.admission),
            "lock-v2 WDL row belongs to another producer",
        )?;
        let record = &mut row.record.0;
        let input = &record.input;
        check(
            record.targets.wdl.is_none()
                && outcome.authority.game_id == input.game_id
                && outcome.authority.anchors.contains(&input.rules_anchor),
            "lock-v2 WDL is not an actual trajectory member",
        )?;
        let sequence = self.original.original.sequence.checked_add(1)
            .ok_or_else(|| invalid("Rovent collector sequence exhausted"))?;
        let raw = json!({"schema":HEAD_DOMAIN,"kind":"rules_wdl","input_sha256":record.input_sha256,
            "anchor":input.rules_anchor,"trajectory":outcome.authority,"rules_receipt":outcome.receipt,"rules_replay":outcome.replay});
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
        input.wdl_probabilities(&target)?;
        let head = RoventHeadSource {
            actor_id: record.lineage.actor_id.clone(),
            source: target.source.clone(),
            binary_sha256: None,
            profile_sha256: None,
            raw_receipt: raw,
            raw_receipt_sha256: raw_sha,
            observed_sequence: sequence,
            anchor: input.rules_anchor.clone(),
        };
        self.original.original.sequence = sequence;
        record.head_sources.insert("wdl".into(), head);
        record.targets.wdl = Some(target);
        Ok(())
    }
}
impl RoventCorpusProducerV2 {
    /// 원 C row를 보존한 caller가 실제 후속 cycle witness를 나중에 붙인다.
    /// 수집기 순번과 manager event 순번을 합치거나 과거 입력을 수정하지 않는다.
    pub fn attach_pc_divergence(
        &mut self,
        row: &mut RoventRetiredRecordV2,
        witness: &rz_search::pals::engine::pc_port::PcDivergenceWitness,
    ) -> Result<(), ArenaError> {
        check(
            row.manifest_sha256 == self.original.manifest_sha256,
            "divergence row belongs to another original corpus owner",
        )?;
        self.validate_admission(&row.admission)?;
        let observed_sequence = self.original.next()?;
        divergence_owner::attach_actual_pc_divergence(row, witness, observed_sequence)
    }
    /// 원 초기 입력과 미래 visible 입력의 실제 퇴역 custody를 함께 받는다.
    /// 최초 입력의 counter=None과 당시 slot은 그대로 두고 target 출처만 잇는다.
    pub fn attach_pc_divergence_from_visible(
        &mut self,
        row: &mut RoventRetiredRecordV2,
        witness: &rz_search::pals::engine::pc_port::PcDivergenceWitness,
        visible: &RoventRetiredRecordV2,
    ) -> Result<(), ArenaError> {
        check(
            row.manifest_sha256 == self.original.manifest_sha256
                && visible.manifest_sha256 == self.original.manifest_sha256
                && row.admission.raw_sha256() == visible.admission.raw_sha256()
                && std::sync::Arc::ptr_eq(
                    row.admission.root_admission(),
                    visible.admission.root_admission(),
                ),
            "future divergence rows belong to another original corpus/root owner",
        )?;
        self.validate_admission(&row.admission)?;
        self.validate_admission(&visible.admission)?;
        let observed_sequence = self.original.next()?;
        divergence_owner::attach_actual_pc_divergence_from_visible(
            row,
            witness,
            visible,
            observed_sequence,
        )
    }
    #[cfg(feature = "pals-collection-onnx")]
    pub(super) fn original_teacher_producer(&self) -> &RoventCorpusProducer {
        &self.original
    }
    pub fn new(manifest: RoventCorpusManifestV2) -> Result<Self, ArenaError> {
        let manifest_sha = manifest.sha256()?;
        let mut original = RoventCorpusProducer::new(manifest.common_registration_projection())?;
        original.manifest_sha256 = manifest_sha;
        original.native_domain = ROVENT_NATIVE_RAW_V3_DOMAIN;
        Ok(Self {
            manifest,
            original,
            inputs: BTreeMap::new(),
        })
    }
    pub fn manifest(&self) -> &RoventCorpusManifestV2 {
        &self.manifest
    }
    pub(super) fn validate_admission(
        &self,
        admission: &RoventCollectionRootAdmission,
    ) -> Result<(), ArenaError> {
        self.original.validate_admission(admission)
    }
    pub fn incomplete_capture_count(&self) -> usize {
        self.original.incomplete_capture_count()
    }
    pub fn submitted_input_seal(
        &self,
        id: RequestId,
    ) -> Result<RoventSubmittedInputSeal, ArenaError> {
        self.original.submitted_input_seal(id)
    }
    pub fn reserve_input(
        &mut self,
        context: GenerationCaptureContextV3,
        lineage: RoventLineage,
    ) -> Result<RoventInputCaptureReservationV2, ArenaError> {
        Ok(RoventInputCaptureReservationV2 {
            original: self.original.reserve_input(context, lineage)?,
            native_domain: self.original.native_domain,
        })
    }
    pub fn admit_captured_input(
        &mut self,
        captured: RoventCapturedInputV2,
    ) -> Result<(), ArenaError> {
        let ticket = captured.original.pending.ticket;
        let key = (ticket.id.epoch.0, ticket.id.sequence);
        check(
            !self.inputs.contains_key(&key) && self.inputs.len() < 16,
            "raw3 original input duplicate/cap",
        )?;
        self.original.admit_captured_input(captured.original)?;
        self.inputs.insert(key, captured.input);
        Ok(())
    }
    pub fn discard_reserved_input(&mut self, sequence: u64) -> bool {
        self.original.discard_reserved_input(sequence)
    }
    pub fn discard_unpublished(&mut self, id: RequestId) -> bool {
        self.inputs.remove(&(id.epoch.0, id.sequence));
        self.original.discard_unpublished(id)
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
        let reservation = self.reserve_input(context, lineage)?;
        let sequence = reservation.sequence();
        let result = reservation
            .capture(ticket, view, input, identity)
            .and_then(|input| self.admit_captured_input(input));
        if result.is_err() {
            self.discard_reserved_input(sequence);
        }
        result
    }
    pub fn observe_native_event<O: ProposerRankingOutput>(
        &mut self,
        event: &TicketNativeEvent<O>,
    ) -> Result<(), ArenaError> {
        self.original.observe_native_event(event)?;
        if let TicketNativeEvent::Retired {
            ticket,
            accepted: false,
            ..
        } = event
        {
            self.inputs.remove(&(ticket.id.epoch.0, ticket.id.sequence));
        }
        Ok(())
    }
    pub fn retire_accepted(
        &mut self,
        admission: &RoventCollectionRootAdmission,
        accepted: &AcceptedRoleStep,
        output: &RoleOutput,
    ) -> Result<RoventRetiredRecordV2, ArenaError> {
        let original = self.original.retire_accepted(admission, accepted, output)?;
        self.join_retired_input(original)
    }
    pub fn retire_root_ranking(
        &mut self,
        accepted: &RoventRootRankingAcceptance,
        output: &RoleOutput,
    ) -> Result<RoventRetiredRecordV2, ArenaError> {
        let original = self.original.retire_root_ranking(accepted, output)?;
        self.join_retired_input(original)
    }
    fn join_retired_input(
        &mut self,
        original: RoventRetiredRecord,
    ) -> Result<RoventRetiredRecordV2, ArenaError> {
        let key = (original.ticket.id.epoch.0, original.ticket.id.sequence);
        let input = self
            .inputs
            .remove(&key)
            .ok_or_else(|| invalid("raw3 original input lost before retirement"))?;
        check(
            input.sha256()? == original.record.input_sha256,
            "raw3 retained input differs from original owner",
        )?;
        Ok(RoventRetiredRecordV2 {
            manifest_sha256: original.manifest_sha256,
            ticket: original.ticket,
            admission: original.admission,
            record: RoventTrainingRecordV2 {
                schema: ROVENT_TRAINING_RECORD_V2_DOMAIN.into(),
                role: original.record.role,
                split: original.record.split,
                input,
                input_sha256: original.record.input_sha256,
                origin_sha256: String::new(),
                record_sha256: String::new(),
                lineage: original.record.lineage,
                native: original.record.native,
                head_sources: BTreeMap::new(),
                targets: RoventTargetsV2::default(),
                divergence_index: None,
                nnuec_input: None,
                nnuec_input_sha256: None,
                nnuec_model_sha256: None,
                nnuec_semantics: None,
                nnuec_wdl: None,
            },
        })
    }
    pub fn publish(
        &self,
        mut row: RoventRetiredRecordV2,
    ) -> Result<PublishedRoventRecordV2, ArenaError> {
        self.seal_retired(&mut row)?;
        Ok(PublishedRoventRecordV2 {
            record: row.record,
            manifest_sha256: row.manifest_sha256,
            admission: row.admission,
        })
    }
    /// 실제 caller가 원 row를 보관한 상태로 게시 전 검증을 수행한다.
    /// 이 계산만으로 PublishedRecord 또는 native 권한을 발급하지 않는다.
    pub(super) fn seal_retired(&self, row: &mut RoventRetiredRecordV2) -> Result<(), ArenaError> {
        check(
            row.manifest_sha256 == self.original.manifest_sha256,
            "raw3 retired row belongs to another owner",
        )?;
        self.validate_admission(&row.admission)?;
        check(
            row.record.native.raw_retirement["root_admission_sha256"] == row.admission.raw_sha256()
                && row.record.native.raw_retirement["root_admission"]
                    == *row.admission.raw_receipt(),
            "raw3 row lost original admission",
        )?;
        row.record.seal(&self.manifest)?;
        Ok(())
    }
    pub fn attach_actual_wdl(
        &mut self,
        row: &mut RoventRetiredRecordV2,
        outcome: &RoventRulesOutcome,
    ) -> Result<(), ArenaError> {
        check(
            row.manifest_sha256 == self.original.manifest_sha256,
            "raw3 WDL row owner mismatch",
        )?;
        let input = &row.record.input;
        check(
            row.record.targets.wdl.is_none()
                && outcome.authority.game_id == input.game_id
                && outcome.authority.anchors.contains(&input.rules_anchor),
            "raw3 WDL is not an actual trajectory member",
        )?;
        let sequence = self.original.next()?;
        let raw = json!({"schema":HEAD_DOMAIN,"kind":"rules_wdl","input_sha256":row.record.input_sha256,
            "anchor":input.rules_anchor,"trajectory":outcome.authority,"rules_receipt":outcome.receipt,"rules_replay":outcome.replay});
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
        input.wdl_probabilities(&target)?;
        row.record.head_sources.insert(
            "wdl".into(),
            RoventHeadSource {
                actor_id: row.record.lineage.actor_id.clone(),
                source: target.source.clone(),
                binary_sha256: None,
                profile_sha256: None,
                raw_receipt: raw,
                raw_receipt_sha256: raw_sha,
                observed_sequence: sequence,
                anchor: input.rules_anchor.clone(),
            },
        );
        row.record.targets.wdl = Some(target);
        Ok(())
    }
}

impl RoventRetiredRecordV2 {
    pub fn input_sha256(&self) -> &str {
        &self.record.input_sha256
    }
    pub fn submitted_input_seal(&self) -> RoventSubmittedInputSeal {
        input_seal(
            &self.manifest_sha256,
            self.ticket,
            &self.record.input.common_validation_projection(),
            &self.record.input_sha256,
            &self.record.lineage,
        )
    }
}

#[cfg(feature = "pals-collection-onnx")]
impl RoventCorpusProducerV2 {
    /// 형식별 입력은 바꾸지 않고 원 dispatch/completion과 policy 계산만 공유한다.
    pub fn attach_collection_teacher_policy(
        &mut self,
        row: &mut RoventRetiredRecordV2,
        binding: RoventCollectionTeacherBinding,
        owner: &rz_uci::rovent_external_launch::RoventCollectionTeacherOwner,
        dispatch: &rz_search::external_cpu::collection_teacher::TeacherDispatch,
        completion: &rz_search::external_cpu::collection_teacher::TeacherCompletion,
    ) -> Result<(), ArenaError> {
        check(
            row.manifest_sha256 == self.original.manifest_sha256,
            "raw3 teacher row belongs to another producer",
        )?;
        self.validate_admission(&row.admission)?;
        let mut projection = RoventRetiredRecord {
            record: historical_v2::common_scope_projection(&row.record),
            manifest_sha256: row.manifest_sha256.clone(),
            ticket: row.ticket,
            admission: row.admission.clone(),
        };
        self.original.attach_collection_teacher_policy(
            &mut projection,
            binding,
            owner,
            dispatch,
            completion,
        )?;
        row.record.targets.policy = projection.record.targets.policy;
        if let Some(source) = projection.record.head_sources.remove("policy") {
            row.record.head_sources.insert("policy".into(), source);
        }
        Ok(())
    }
}
