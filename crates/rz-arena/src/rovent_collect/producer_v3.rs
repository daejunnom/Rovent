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
    // A prepared target remains opaque and unpublished until the same
    // original ticket/root publication gate has been acquired.
    prepared_policy: Option<super::teacher::policy_prepare_v3::PreparedTeacherPolicyLockV3>,
}
impl RoventRetiredLockV3Record {
    /// Borrow the original accepted row for target preparation. This does not
    /// transfer producer custody, publish the row, or authorize a target.
    pub(super) fn record(
        &self,
    ) -> &rz_experiments::rovent_corpus_lock_v3::record::RoventTrainingRecordLockV3 {
        &self.record
    }
    pub(super) fn admission(&self) -> &RoventCollectionRootAdmissionV3 {
        &self.admission
    }
    pub(super) fn ticket(&self) -> RoleTicketEnvelope {
        self.ticket
    }
    pub(super) fn manifest(&self) -> &Arc<RoventCorpusManifestLockV3> {
        &self.manifest_owner
    }
    /// Recover exactly the staged owner on cancel or an unsupported writer.
    /// Taking it does not issue target, publication or completion authority.
    pub(super) fn take_prepared_teacher_policy(
        &mut self,
    ) -> Option<super::teacher::policy_prepare_v3::PreparedTeacherPolicyLockV3> {
        self.prepared_policy.take()
    }
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
            prepared_policy: None,
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
            row.prepared_policy.is_none()
                && targets.policy.is_none()
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

    /// Retain the original prepared teacher evidence with this exact
    /// producer's retired row. This does not commit a target or bypass the
    /// evidence-only publisher. A future gated writer must establish the
    /// cancellation/publication ordering before consuming the prepared owner.
    pub(super) fn stage_prepared_teacher_policy(
        &self,
        row: &mut RoventRetiredLockV3Record,
        prepared: &mut Option<super::teacher::policy_prepare_v3::PreparedTeacherPolicyLockV3>,
    ) -> Result<(), ArenaError> {
        let value = prepared
            .as_ref()
            .ok_or_else(|| invalid("lock-v3 prepared teacher custody is empty"))?;
        check(
            row.prepared_policy.is_none()
                && row.record.0.targets.policy.is_none()
                && !row.record.0.head_sources.contains_key("policy")
                && Arc::ptr_eq(&row.producer_owner, &self.producer_owner)
                && Arc::ptr_eq(&row.manifest_owner, &self.manifest)
                && row.admission.same_original_admission(&self.admission)
                && value.ticket() == row.ticket
                && value
                    .sealed()
                    .evidence()
                    .admission()
                    .same_original_admission(&self.admission)
                && !value.sealed().capsule().is_ticket_canceled(),
            "lock-v3 prepared policy differs from original producer/root or is canceled",
        )?;
        value.sealed().capsule().matches_original_row(row)?;
        // No fallible work between transfer and installation in the original
        // row: the caller retains its option on every earlier rejection.
        row.prepared_policy = prepared.take();
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

const TICKET_ISSUING: u8 = 1;
const TICKET_CANCELED: u8 = 2;
const TICKET_ADMITTED: u8 = 4;
const TICKET_ISSUE_UNKNOWN: u8 = 8;

#[derive(Clone)]
pub(crate) struct SpeculativeTicketAuthorityV3(std::sync::Arc<std::sync::atomic::AtomicU8>);
impl SpeculativeTicketAuthorityV3 {
    pub(crate) fn new() -> Self {
        Self(std::sync::Arc::new(std::sync::atomic::AtomicU8::new(0)))
    }
    /// Return whether an issuance claim was already acquired. This is not
    /// proof of actual mailbox admission or physical completion.
    pub(crate) fn cancel(&self) -> bool {
        self.0
            .fetch_or(TICKET_CANCELED, std::sync::atomic::Ordering::AcqRel)
            & TICKET_ISSUING
            != 0
    }
    pub(super) fn canceled(&self) -> bool {
        self.0.load(std::sync::atomic::Ordering::Acquire) & TICKET_CANCELED != 0
    }
    fn claim(&self) -> Result<SpeculativeIssuanceClaimV3, ArenaError> {
        self.0
            .compare_exchange(
                0,
                TICKET_ISSUING,
                std::sync::atomic::Ordering::AcqRel,
                std::sync::atomic::Ordering::Acquire,
            )
            .map_err(|_| {
                invalid("speculative ticket issuance canceled, already claimed, or unknown")
            })?;
        Ok(SpeculativeIssuanceClaimV3 {
            authority: self.clone(),
            resolved: false,
        })
    }
}
/// One claimed ticket immediately surrounding actual mailbox admission. The
/// owner must mark the actual result. Unwind/unknown blocks retries.
pub(crate) struct SpeculativeIssuanceClaimV3 {
    authority: SpeculativeTicketAuthorityV3,
    resolved: bool,
}
impl SpeculativeIssuanceClaimV3 {
    /// Called only on the original mailbox's successful admission result.
    pub(crate) fn admitted(mut self) {
        self.authority
            .0
            .fetch_or(TICKET_ADMITTED, std::sync::atomic::Ordering::AcqRel);
        self.resolved = true;
    }
    /// Called only when original mailbox admission is known not to have occurred.
    pub(crate) fn rejected(mut self) {
        self.authority
            .0
            .fetch_and(!TICKET_ISSUING, std::sync::atomic::Ordering::AcqRel);
        self.resolved = true;
    }
}
impl Drop for SpeculativeIssuanceClaimV3 {
    fn drop(&mut self) {
        if !self.resolved {
            self.authority
                .0
                .fetch_or(TICKET_ISSUE_UNKNOWN, std::sync::atomic::Ordering::AcqRel);
        }
    }
}

fn claim_speculative_issuance_v3(
    root_revoked: &std::sync::atomic::AtomicBool,
    root_canceled: &std::sync::atomic::AtomicBool,
    root_targets_revoked: &std::sync::atomic::AtomicBool,
    ticket: &SpeculativeTicketAuthorityV3,
) -> Result<SpeculativeIssuanceClaimV3, ArenaError> {
    use std::sync::atomic::Ordering;
    root_revoked
        .compare_exchange(false, false, Ordering::AcqRel, Ordering::Acquire)
        .map_err(|_| invalid("speculative root is revoked"))?;
    root_targets_revoked
        .compare_exchange(false, false, Ordering::AcqRel, Ordering::Acquire)
        .map_err(|_| invalid("speculative root targets are revoked"))?;
    root_canceled
        .compare_exchange(false, false, Ordering::AcqRel, Ordering::Acquire)
        .map_err(|_| invalid("speculative root teacher admission is closed"))?;
    ticket.claim()
}

/// Original native submission evidence for an optional speculative teacher.
/// It is not NN acceptance, a retired record, a target, or a completion proof.
/// Construction is restricted to the same private capture and root owner.
pub(crate) struct SubmittedTeacherQueryV3 {
    admission: rz_uci::pals_adaptive::RoventCollectionRootAdmissionV3,
    producer_owner: std::sync::Arc<()>,
    input: RoventSubmittedInputSeal,
    anchor: rz_position::PositionSnapshot,
    prefix: Vec<rz_position::BoardMove>,
    submitted_prefix: Vec<u16>,
    legal: Vec<u16>,
    scalar_only: bool,
    ticket_authority: SpeculativeTicketAuthorityV3,
    native_cancellation: rz_uci::pals_native::ticketed::NativeTicketCancelObservation,
    root_revoked: std::sync::Arc<std::sync::atomic::AtomicBool>,
    root_canceled: std::sync::Arc<std::sync::atomic::AtomicBool>,
    root_targets_revoked: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl SubmittedTeacherQueryV3 {
    pub(crate) fn input_seal(&self) -> RoventSubmittedInputSeal {
        RoventSubmittedInputSeal {
            manifest_sha256: self.input.manifest_sha256.clone(),
            ticket: self.input.ticket,
            input_sha256: self.input.input_sha256.clone(),
            selected_view_sha256: self.input.selected_view_sha256.clone(),
            anchor: self.input.anchor.clone(),
            role: self.input.role,
            teacher_actor_id: self.input.teacher_actor_id.clone(),
        }
    }
    pub(crate) fn admission(&self) -> &rz_uci::pals_adaptive::RoventCollectionRootAdmissionV3 {
        &self.admission
    }
    pub(crate) fn ticket(&self) -> RoleTicketEnvelope {
        self.input.ticket
    }
    pub(crate) fn has_policy_candidates(&self) -> bool {
        !self.legal.is_empty()
    }
    pub(crate) fn requires_teacher(&self) -> bool {
        self.input.teacher_actor_id.is_some() && self.has_policy_candidates()
    }
    /// Logical requests only; neither proves worker or device completion.
    /// Root closure stops new submissions, while an already sealed result may
    /// still join its valid original retirement if its ticket was not revoked.
    pub(crate) fn is_ticket_canceled(&self) -> bool {
        self.ticket_authority.canceled()
            || self
                .root_targets_revoked
                .load(std::sync::atomic::Ordering::Acquire)
    }
    pub(crate) fn is_canceled(&self) -> bool {
        self.is_ticket_canceled()
            || self.native_cancellation.is_canceled()
            || self.root_revoked.load(std::sync::atomic::Ordering::Acquire)
            || self
                .root_canceled
                .load(std::sync::atomic::Ordering::Acquire)
    }
    /// Observe cancellation and claim this exact ticket once. The independent
    /// root/native observations and ticket CAS are not one atomic gate. A
    /// concurrent later cancellation requires the mailbox owner to retain,
    /// cancel and drain any admitted or unresolved teacher work.
    pub(crate) fn try_claim_teacher_issuance(
        &self,
    ) -> Result<SpeculativeIssuanceClaimV3, ArenaError> {
        // This CAS is against the exact native preparation cancellation
        // word, not a later observer projection. It authorizes no NN result.
        check(
            self.native_cancellation.try_claim_submission(),
            "original native ticket is canceled",
        )?;
        let claim = claim_speculative_issuance_v3(
            &self.root_revoked,
            &self.root_canceled,
            &self.root_targets_revoked,
            &self.ticket_authority,
        )?;
        if self.is_canceled() {
            claim.rejected();
            return Err(invalid(
                "speculative query canceled before mailbox submission",
            ));
        }
        Ok(claim)
    }

    /// Reconstruct and verify outside the native callback. This does not
    /// acquire live-root, acceptance, execution or publication authority.
    pub(crate) fn original_teacher_query_position(
        &self,
    ) -> Result<rz_position::Position, ArenaError> {
        let mut position = rz_position::Position::from_snapshot(
            &self.anchor,
            rz_position::PositionLimits {
                max_history_positions: 16_385,
                ..Default::default()
            },
        )
        .map_err(|error| invalid(error.to_string()))?;
        check(
            self.prefix.len() <= 256 && packed(&self.prefix)? == self.submitted_prefix,
            "speculative query prefix differs from original submitted input",
        )?;
        for movement in &self.prefix {
            position
                .make_move(*movement)
                .map_err(|error| invalid(error.to_string()))?;
        }
        check(
            exact_rules_anchor(&position)? == self.input.anchor,
            "speculative query Rules anchor differs",
        )?;
        let legal = packed(&position.legal_moves())?;
        if self.scalar_only {
            check(
                self.legal.is_empty(),
                "speculative scalar query has candidates",
            )?;
        } else {
            check(
                legal.len() == self.legal.len()
                    && self.legal.iter().all(|item| legal.contains(item)),
                "speculative query candidates differ from Rules",
            )?;
        }
        Ok(position)
    }
    /// Compare identity already retained by the original producer. This is
    /// not a retirement witness or physical/target authority.
    pub(crate) fn matches_original_identity(
        &self,
        admission: &rz_uci::pals_adaptive::RoventCollectionRootAdmissionV3,
        producer_owner: &std::sync::Arc<()>,
        original: &RoventSubmittedInputSeal,
    ) -> Result<(), ArenaError> {
        check(
            std::sync::Arc::ptr_eq(&self.producer_owner, producer_owner)
                && self.admission.same_original_admission(admission)
                && self.input.manifest_sha256 == original.manifest_sha256
                && self.input.ticket == original.ticket
                && self.input.input_sha256 == original.input_sha256
                && self.input.selected_view_sha256 == original.selected_view_sha256
                && self.input.anchor == original.anchor
                && self.input.role == original.role
                && self.input.teacher_actor_id == original.teacher_actor_id,
            "speculative teacher query differs from retained original identity",
        )
    }
    /// Only the actual producer-retired row can join an earlier query. This
    /// comparison grants no native fence or label authority by itself.
    pub(super) fn matches_original_row(
        &self,
        row: &RoventRetiredLockV3Record,
    ) -> Result<(), ArenaError> {
        let record = &row.record.0;
        check(
            std::sync::Arc::ptr_eq(&self.producer_owner, &row.producer_owner)
                && self.admission.same_original_admission(&row.admission)
                && self.input.manifest_sha256 == row.manifest_sha256
                && self.input.ticket == row.ticket
                && self.input.input_sha256 == record.input_sha256
                && self.input.selected_view_sha256 == record.input.view_sha256
                && self.input.anchor == record.input.rules_anchor
                && self.input.role == record.input.role
                && self.input.teacher_actor_id == record.lineage.teacher_actor_id,
            "speculative teacher query differs from the original retired owner",
        )
    }
}

impl RoventCapturedInputV3 {
    pub(super) fn speculative_teacher_required(&self) -> bool {
        self.original
            .original
            .pending
            .lineage
            .teacher_actor_id
            .is_some()
            && !self.original.original.pending.input.legal_moves.is_empty()
    }
    /// Build only bounded immutable capture from the exact same submitted
    /// ticket. Expensive replay and legal generation happen on the worker.
    pub(super) fn speculative_teacher_query(
        &self,
        admission: rz_uci::pals_adaptive::RoventCollectionRootAdmissionV3,
        view: &ReviewView,
        input: &PalsVirtualReviewInput,
        ticket_authority: SpeculativeTicketAuthorityV3,
        native_cancellation: rz_uci::pals_native::ticketed::NativeTicketCancelObservation,
        root_revoked: std::sync::Arc<std::sync::atomic::AtomicBool>,
        root_canceled: std::sync::Arc<std::sync::atomic::AtomicBool>,
        root_targets_revoked: std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) -> Result<SubmittedTeacherQueryV3, ArenaError> {
        check(
            self.admission.same_original_admission(&admission),
            "V3 teacher query admission differs from original capture",
        )?;
        let owner = self
            .original
            .producer_owner
            .as_ref()
            .ok_or_else(|| invalid("speculative producer owner missing"))?;
        let anchor = view
            .anchors()
            .get(usize::from(input.query.anchors[0]))
            .ok_or_else(|| invalid("speculative query anchor missing"))?
            .snapshot
            .clone();
        let mut prefix = Vec::new();
        prefix
            .try_reserve_exact(view.query().prefix.len())
            .map_err(|_| invalid("speculative prefix allocation"))?;
        prefix.extend_from_slice(&view.query().prefix);
        let original = &self.original.original.pending.input;
        let mut submitted_prefix = Vec::new();
        submitted_prefix
            .try_reserve_exact(original.selected_view.query.prefix.len())
            .map_err(|_| invalid("speculative submitted prefix allocation"))?;
        submitted_prefix.extend_from_slice(&original.selected_view.query.prefix);
        let mut legal = Vec::new();
        legal
            .try_reserve_exact(original.legal_moves.len())
            .map_err(|_| invalid("speculative candidates allocation"))?;
        legal.extend_from_slice(&original.legal_moves);
        let scalar_only = matches!(
            original.selected_view.query.purpose.as_str(),
            "divergence_policy" | "task_policy"
        );
        Ok(SubmittedTeacherQueryV3 {
            admission,
            producer_owner: std::sync::Arc::clone(owner),
            input: self.original.submitted_input_seal(),
            anchor,
            prefix,
            submitted_prefix,
            legal,
            scalar_only,
            ticket_authority,
            native_cancellation,
            root_revoked,
            root_canceled,
            root_targets_revoked,
        })
    }
}
#[cfg(test)]
mod v3_issuance_tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn cancel_and_unknown_do_not_reissue_a_ticket() {
        let roots = AtomicBool::new(false);
        let ticket = SpeculativeTicketAuthorityV3::new();
        let claim = claim_speculative_issuance_v3(&roots, &roots, &roots, &ticket).unwrap();
        assert!(ticket.cancel());
        drop(claim);
        assert!(claim_speculative_issuance_v3(&roots, &roots, &roots, &ticket).is_err());

        let unknown = SpeculativeTicketAuthorityV3::new();
        drop(claim_speculative_issuance_v3(&roots, &roots, &roots, &unknown).unwrap());
        assert!(claim_speculative_issuance_v3(&roots, &roots, &roots, &unknown).is_err());
    }

    #[test]
    fn known_rejection_may_retry_only_while_root_open() {
        let root = AtomicBool::new(false);
        let canceled = AtomicBool::new(false);
        let targets = AtomicBool::new(false);
        let ticket = SpeculativeTicketAuthorityV3::new();
        claim_speculative_issuance_v3(&root, &canceled, &targets, &ticket)
            .unwrap()
            .rejected();
        let claim = claim_speculative_issuance_v3(&root, &canceled, &targets, &ticket).unwrap();
        root.store(true, Ordering::Release);
        claim.rejected();
        assert!(claim_speculative_issuance_v3(&root, &canceled, &targets, &ticket).is_err());
    }
}
