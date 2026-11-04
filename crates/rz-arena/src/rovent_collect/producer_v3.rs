//! System-runtime /3 collection owner. Common submitted-input payload encoding
//! is reused; live admission, tickets, witnesses and publication remain /3.
//! This producer currently emits evidence-only P/C rows: policy, WDL,
//! divergence and task targets are absent. Such rows do not establish valid
//! supervised targets or eligibility for a training generation.
use super::producer_v2::{
    RoventCapturedInputV2, RoventCorpusProducerV2, RoventInputCaptureReservationV2,
};
use super::*;
use rz_experiments::rovent_corpus_lock_v3::record::RoventCorpusManifestLockV3;
use rz_experiments::rovent_corpus_v2::{RoventTargetsV2, RoventTrainingRecordV2};
use rz_uci::pals_adaptive::RoventCollectionRootAdmissionV3;
use std::sync::Arc;

pub(super) struct RoventInputCaptureReservationV3 {
    original: RoventInputCaptureReservationV2,
    admission: RoventCollectionRootAdmissionV3,
    ticket: RoleTicketEnvelope,
}
pub(super) struct RoventCapturedInputV3 {
    original: RoventCapturedInputV2,
    admission: RoventCollectionRootAdmissionV3,
    ticket: RoleTicketEnvelope,
}
impl RoventInputCaptureReservationV3 {
    pub(super) fn sequence(&self) -> u64 {
        self.original.sequence()
    }
    pub(super) fn ticket(&self) -> &RoleTicketEnvelope {
        &self.ticket
    }
    pub(super) fn admission(&self) -> &RoventCollectionRootAdmissionV3 {
        &self.admission
    }
    pub(super) fn capture(
        self,
        ticket: RoleTicketEnvelope,
        view: &ReviewView,
        input: &PalsVirtualReviewInput,
        identity: &ReviewExecutionIdentity,
    ) -> Result<RoventCapturedInputV3, Box<(ArenaError, Self)>> {
        if self.ticket != ticket {
            return Err(Box::new((
                invalid("lock-v3 capture ticket differs from original reservation"),
                self,
            )));
        }
        let Self {
            original,
            admission,
            ticket,
        } = self;
        match original.try_capture_preserving_owner(ticket, view, input, identity) {
            Ok(original) => Ok(RoventCapturedInputV3 {
                original,
                admission,
                ticket,
            }),
            Err(failure) => {
                let (error, original) = *failure;
                Err(Box::new((
                    error,
                    Self {
                        original,
                        admission,
                        ticket,
                    },
                )))
            }
        }
    }
}
impl RoventCapturedInputV3 {
    pub(super) fn admission(&self) -> &RoventCollectionRootAdmissionV3 {
        &self.admission
    }
    pub(super) fn ticket(&self) -> &RoleTicketEnvelope {
        &self.ticket
    }
}
pub(super) struct RoventRetiredLockV3Record {
    record: rz_experiments::rovent_corpus_lock_v3::record::RoventTrainingRecordLockV3,
    manifest_sha256: String,
    manifest_owner: Arc<RoventCorpusManifestLockV3>,
    producer_owner: Arc<()>,
    admission: RoventCollectionRootAdmissionV3,
    ticket: RoleTicketEnvelope,
}
pub(super) struct RoventCorpusProducerLockV3 {
    manifest: Arc<RoventCorpusManifestLockV3>,
    admission: RoventCollectionRootAdmissionV3,
    producer_owner: Arc<()>,
    original: RoventCorpusProducerV2,
    native_retirements: BTreeMap<RequestId, RoleTicketEnvelope>,
    native_observation_failures:
        BTreeMap<RequestId, Vec<rz_uci::pals_native::ticketed::TicketNativeObservationFailure>>,
}
impl RoventCorpusProducerLockV3 {
    pub(super) fn new_shared(
        manifest: Arc<RoventCorpusManifestLockV3>,
        admission: RoventCollectionRootAdmissionV3,
    ) -> Result<Self, ArenaError> {
        let lock = admission.native_origin().original_input_lock();
        manifest.validate(lock)?;
        rz_experiments::rovent_corpus_lock_v3::native::validate_lock_v3_root_metadata(
            admission.raw_receipt(),
            &json!(admission.raw_sha256()),
            lock,
        )?;
        let manifest_sha256 = manifest.sha256(lock)?;
        let mut original = RoventCorpusProducerV2::new(manifest.registration.clone())?;
        original.original.manifest_sha256 = manifest_sha256;
        original.original.native_domain =
            rz_experiments::rovent_corpus_lock_v3::native::ROVENT_LOCK_V3_NATIVE_RAW_DOMAIN;
        Ok(Self {
            manifest,
            admission,
            producer_owner: Arc::new(()),
            original,
            native_retirements: BTreeMap::new(),
            native_observation_failures: BTreeMap::new(),
        })
    }
    pub(super) fn validate_admission(
        &self,
        admission: &RoventCollectionRootAdmissionV3,
    ) -> Result<(), ArenaError> {
        check(
            self.admission.same_original_admission(admission),
            "lock-v3 collection admission differs from original owner",
        )
    }
    pub(super) fn try_reserve_input(
        &mut self,
        ticket: &RoleTicketEnvelope,
        context: GenerationCaptureContextV3,
        lineage: RoventLineage,
    ) -> Result<Option<RoventInputCaptureReservationV3>, ArenaError> {
        check(
            matches!(ticket.role, Role::Proposer | Role::Critic),
            "lock-v3 P/C evidence producer does not support validator tickets",
        )?;
        if let Err(error) = self.admission.try_validate_ticket(ticket) {
            return if error.kind == rz_uci::pals_adaptive::RoventTicketValidationKindV2::Busy {
                Ok(None)
            } else {
                Err(invalid(error.error.to_string()))
            };
        }
        let mut original = self.original.reserve_input(context, lineage)?;
        original.producer_owner = Some(Arc::clone(&self.producer_owner));
        Ok(Some(RoventInputCaptureReservationV3 {
            original,
            admission: self.admission.clone(),
            ticket: *ticket,
        }))
    }
    pub(super) fn admit_captured_input(
        &mut self,
        captured: RoventCapturedInputV3,
    ) -> Result<(), Box<(ArenaError, RoventCapturedInputV3)>> {
        let validation = (|| {
            self.validate_admission(&captured.admission)?;
            let input = &captured.original;
            let key = (captured.ticket.id.epoch.0, captured.ticket.id.sequence);
            let owner = &self.original.original;
            check(
                input.original.pending.ticket == captured.ticket
                    && input
                        .producer_owner
                        .as_ref()
                        .is_some_and(|a| Arc::ptr_eq(a, &self.producer_owner))
                    && input.original.manifest_sha256 == owner.manifest_sha256
                    && owner.reserved.contains(&input.original.pending.submitted)
                    && !owner.pending.contains_key(&key)
                    && !owner.retired.contains(&key)
                    && !self.original.inputs.contains_key(&key),
                "lock-v3 captured input differs from original owner, reservation, or ticket",
            )
        })();
        if let Err(error) = validation {
            return Err(Box::new((error, captured)));
        }
        let input = captured.original;
        let key = (captured.ticket.id.epoch.0, captured.ticket.id.sequence);
        self.original
            .original
            .reserved
            .remove(&input.original.pending.submitted);
        self.original
            .original
            .pending
            .insert(key, input.original.pending);
        self.original.inputs.insert(key, input.input);
        Ok(())
    }
    pub(super) fn discard_reserved_input(&mut self, sequence: u64) -> bool {
        self.original.discard_reserved_input(sequence)
    }
    pub(super) fn incomplete_capture_count(&self) -> usize {
        self.original.incomplete_capture_count()
    }
    /// Observation errors are scoped to this ticket and remain distinct from
    /// the worker's physical-completion and owner-fence evidence.
    pub(super) fn native_observation_failures(
        &self,
        id: RequestId,
    ) -> Option<&[rz_uci::pals_native::ticketed::TicketNativeObservationFailure]> {
        self.native_observation_failures.get(&id).map(Vec::as_slice)
    }
    pub fn observe_native_event<O: ProposerRankingOutput>(
        &mut self,
        witness: rz_uci::pals_adaptive::RoventCollectionNativeEventWitnessV3<'_, O>,
    ) -> Result<(), ArenaError> {
        check(
            self.admission.same_original_admission(witness.admission()),
            "lock-v3 native event belongs to another original admission",
        )?;
        let event = witness.event();
        let (ticket, failures) = match event {
            TicketNativeEvent::Completed {
                completion,
                observation_failures,
                ..
            } => (Some(completion.ticket), Some(observation_failures)),
            TicketNativeEvent::Retired {
                ticket,
                observation_failures,
                ..
            } => (Some(*ticket), Some(observation_failures)),
            _ => (None, None),
        };
        if let (Some(ticket), Some(failures)) = (ticket, failures) {
            if !failures.is_empty() {
                check(
                    self.native_observation_failures.len() < MAX_CORPUS_ROWS
                        || self.native_observation_failures.contains_key(&ticket.id),
                    "lock-v3 native failure evidence cap",
                )?;
                let records = self
                    .native_observation_failures
                    .entry(ticket.id)
                    .or_default();
                records.extend(failures.iter().cloned());
                return Err(invalid(format!(
                    "lock-v3 original native observation failed: {records:?}"
                )));
            }
        }
        self.original.observe_native_event(event)?;
        if let TicketNativeEvent::Retired {
            ticket,
            accepted: true,
            result: Ok(()),
            ..
        } = event
        {
            check(
                !self.native_observation_failures.contains_key(&ticket.id)
                    && self.native_retirements.len() < MAX_CORPUS_ROWS
                    && !self.native_retirements.contains_key(&ticket.id),
                "lock-v3 duplicate or failed native retirement",
            )?;
            self.native_retirements.insert(ticket.id, *ticket);
        }
        Ok(())
    }

    pub fn discard_unpublished(&mut self, id: RequestId) -> bool {
        let discarded = self.original.discard_unpublished(id);
        discarded
    }

    /// Consumes the manager-issued witness for an actually accepted and
    /// search-consumed result. The pending service completion is checked
    /// separately; the witness itself is not a physical-completion claim.
    pub fn retire_consumed(
        &mut self,
        witness: rz_uci::pals_adaptive::RoventCollectionConsumedWitnessV3<'_>,
    ) -> Result<RoventRetiredLockV3Record, ArenaError> {
        check(
            self.admission.same_original_admission(witness.admission()),
            "lock-v3 consumption belongs to another original admission",
        )?;
        let accepted = witness.accepted();
        self.retire_original(
            *accepted.ticket(),
            accepted.output_digest(),
            "accepted_role_step",
        )
    }

    /// The manager's original root-ranking publication is a separate proof;
    /// it must match this owner and the actual native output and completion.
    pub fn retire_root_ranking(
        &mut self,
        accepted: &rz_uci::pals_adaptive::RoventRootRankingAcceptanceV3,
        output: &RoleOutput,
    ) -> Result<RoventRetiredLockV3Record, ArenaError> {
        check(
            self.admission.same_original_admission(accepted.admission()),
            "lock-v3 ranking belongs to another original admission",
        )?;
        let source = accepted.ranking().source();
        accepted
            .admission()
            .validate_output(&source.ticket, output)
            .map_err(|error| invalid(error.to_string()))?;
        check(
            native_acceptance_output_digest(output).map_err(|error| invalid(error.to_string()))?
                == source.output_digest,
            "lock-v3 original ranking/output digest mismatch",
        )?;
        self.retire_original(
            source.ticket,
            source.output_digest,
            "root_ranking_published",
        )
    }

    fn retire_original(
        &mut self,
        ticket: RoleTicketEnvelope,
        output_digest: Digest,
        acceptance_kind: &str,
    ) -> Result<RoventRetiredLockV3Record, ArenaError> {
        check(
            self.native_retirements.get(&ticket.id) == Some(&ticket)
                && !self.native_observation_failures.contains_key(&ticket.id),
            "lock-v3 original native retirement is missing or reported an observation failure",
        )?;
        let key = (ticket.id.epoch.0, ticket.id.sequence);
        let producer = &mut self.original.original;
        let pending = producer
            .pending
            .get(&key)
            .ok_or_else(|| invalid("lock-v3 retirement has no original input"))?;
        let completed = pending
            .completed
            .as_ref()
            .ok_or_else(|| invalid("lock-v3 retirement before physical completion"))?;
        let input = self
            .original
            .inputs
            .get(&key)
            .ok_or_else(|| invalid("lock-v3 retained input is missing"))?;
        check(
            pending.ticket == ticket
                && completed.output_digest == hex(output_digest)
                && input.sha256()? == pending.input_sha256
                && !producer.retired.contains(&key)
                && producer.retired.len() < MAX_CORPUS_ROWS,
            "lock-v3 accepted result differs from original physical completion",
        )?;
        let split = *producer
            .manifest
            .opening_splits
            .get(&input.opening_id)
            .ok_or_else(|| invalid("lock-v3 opening split is not registered"))?;
        let sequence = producer
            .sequence
            .checked_add(1)
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
        Ok(RoventRetiredLockV3Record {
            manifest_sha256: producer.manifest_sha256.clone(),
            manifest_owner: std::sync::Arc::clone(&self.manifest),
            producer_owner: std::sync::Arc::clone(&self.producer_owner),
            ticket,
            admission: self.admission.clone(),
            record: rz_experiments::rovent_corpus_lock_v3::record::RoventTrainingRecordLockV3(
                RoventTrainingRecordV2 {
                    schema: rz_experiments::rovent_corpus_lock_v3::record::ROVENT_TRAINING_RECORD_LOCK_V3_DOMAIN.into(),
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
    pub fn seal_retired(&self, row: &mut RoventRetiredLockV3Record) -> Result<(), ArenaError> {
        let targets = &row.record.0.targets;
        check(
            targets.policy.is_none()
                && targets.wdl.is_none()
                && targets.divergence.is_none()
                && targets.task.is_none()
                && row.record.0.head_sources.is_empty(),
            "lock-v3 evidence-only producer cannot issue supervised targets",
        )?;
        check(
            std::sync::Arc::ptr_eq(&row.producer_owner, &self.producer_owner)
                && std::sync::Arc::ptr_eq(&row.manifest_owner, &self.manifest)
                && row.manifest_sha256 == self.original.original.manifest_sha256
                && row.admission.same_original_admission(&self.admission)
                && row.record.0.native.raw_retirement["root_admission_sha256"]
                    == self.admission.raw_sha256()
                && row.record.0.native.raw_retirement["root_admission"]
                    == *self.admission.raw_receipt(),
            "lock-v3 retired record lost original owner",
        )?;
        let mut candidate = row.record.clone();
        candidate.seal(
            &self.manifest,
            self.admission.native_origin().original_input_lock(),
        )?;
        row.record = candidate;
        Ok(())
    }

    /// Publication of the in-process row does not write files. On failure the
    /// caller gets the original row and its admission back for recovery.
    pub fn publish(
        &self,
        mut row: RoventRetiredLockV3Record,
    ) -> Result<PublishedRoventLockV3Record, Box<(ArenaError, RoventRetiredLockV3Record)>> {
        if let Err(error) = self.seal_retired(&mut row) {
            return Err(Box::new((error, row)));
        }
        Ok(PublishedRoventLockV3Record {
            record: row.record,
            manifest_sha256: row.manifest_sha256,
            admission: row.admission,
        })
    }
}
