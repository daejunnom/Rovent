//! Linear acceptance permits and private native-owner success receipts.
//! The actual service creates its issuer on the owner thread, publishes only
//! the verifier, and issues receipts only after backend acceptance succeeds.
use super::{CoordinatorError, RoleTicketRegistry, TicketPhase};
use rz_contracts::pals::{Role, RoleOutput, RolePayload};
use rz_contracts::pals_adaptive::RoleTicketEnvelope;
use rz_contracts::{ClockDomain, Digest, MonotonicTick, Wdl};
use sha2::{Digest as _, Sha256};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct NativeAcceptanceSeal {
    ticket: RoleTicketEnvelope,
    output_digest: Digest,
    reciprocal: bool,
}

/// Registry-issued, non-Clone authority to ask the exact native owner to
/// accept one original P/V output. A permit by itself is never success proof.
#[derive(Debug)]
pub struct NativeAcceptancePermit {
    seal: NativeAcceptanceSeal,
}
impl NativeAcceptancePermit {
    pub fn ticket(&self) -> &RoleTicketEnvelope {
        &self.seal.ticket
    }
    pub fn output_digest(&self) -> Digest {
        self.seal.output_digest
    }
}

/// Hold exclusively inside the actual native owner. Other owner pairs have
/// distinct private identities and cannot confirm this registry's requests.
pub struct NativeAcceptanceIssuer {
    identity: Arc<()>,
}
#[derive(Clone, Debug)]
pub struct NativeAcceptanceVerifier {
    identity: Arc<()>,
}
impl NativeAcceptanceVerifier {
    /// Compare the original native issuer identity, without issuing a permit
    /// or asserting liveness, output acceptance, or physical completion.
    pub fn same_owner(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.identity, &other.identity)
    }
}
impl NativeAcceptanceIssuer {
    pub fn new_pair() -> (Self, NativeAcceptanceVerifier) {
        let identity = Arc::new(());
        (
            Self {
                identity: Arc::clone(&identity),
            },
            NativeAcceptanceVerifier { identity },
        )
    }
    /// The native service calls this only after exact held ticket/output digest
    /// validation and successful backend.accept; failure never issues a receipt.
    pub fn issue_success(&self, permit: NativeAcceptancePermit) -> NativeAcceptanceReceipt {
        NativeAcceptanceReceipt {
            permit,
            identity: Arc::clone(&self.identity),
        }
    }
}

#[cfg(test)]
mod owner_identity_tests {
    use super::NativeAcceptanceIssuer;

    #[test]
    fn verifier_owner_identity_tracks_clones_not_distinct_issuers() {
        let (_first_issuer, first) = NativeAcceptanceIssuer::new_pair();
        let (_second_issuer, second) = NativeAcceptanceIssuer::new_pair();
        assert!(first.same_owner(&first.clone()));
        assert!(!first.same_owner(&second));
        assert!(!second.same_owner(&first));
    }
}

#[derive(Debug)]
pub struct NativeAcceptanceReceipt {
    permit: NativeAcceptancePermit,
    identity: Arc<()>,
}
impl NativeAcceptanceReceipt {
    pub fn ticket(&self) -> &RoleTicketEnvelope {
        self.permit.ticket()
    }
    pub fn output_digest(&self) -> Digest {
        self.permit.output_digest()
    }
}
#[derive(Debug)]
pub struct RejectedNativeAcceptance {
    pub error: CoordinatorError,
    pub receipt: NativeAcceptanceReceipt,
}

/// Successful original P/C native acceptance and registry consumption. Public
/// RoleAcceptance descriptions and a caller boolean cannot construct this seal.
#[derive(Debug)]
pub struct AcceptedRoleStep {
    ticket: RoleTicketEnvelope,
    output_digest: Digest,
}
impl AcceptedRoleStep {
    pub fn ticket(&self) -> &RoleTicketEnvelope {
        &self.ticket
    }
    pub fn output_digest(&self) -> Digest {
        self.output_digest
    }
    pub fn validates_output(&self, output: &RoleOutput) -> bool {
        output.id == self.ticket.id
            && output.authority == self.ticket.authority
            && output.key == self.ticket.representation
            && native_acceptance_output_digest(output).ok() == Some(self.output_digest)
    }

    /// A local accepted checkpoint is not a fictitious P->P or R->P message.
    /// Only an actual peer envelope may populate QuestionEnvelope.parent.
    pub fn origin(
        &self,
        frame: super::FrameId,
        peer: Option<rz_contracts::rovent::QuestionEnvelope>,
    ) -> Result<AcceptedRoleOrigin, CoordinatorError> {
        if frame.authority != self.ticket.authority || frame.sequence == 0 {
            return Err(CoordinatorError::IdentityMismatch("accepted source frame"));
        }
        if let Some(peer) = peer {
            let recipient = match self.ticket.role {
                Role::Proposer => rz_contracts::rovent::SearchRecipient::Proposer,
                Role::Critic => rz_contracts::rovent::SearchRecipient::Critic,
                Role::Validator => {
                    return Err(CoordinatorError::InvalidTransition(
                        "V is not a reciprocal source",
                    ));
                }
            };
            if peer.id.authority != self.ticket.authority
                || peer.question_digest != self.ticket.question
                || peer.view != self.ticket.view
                || peer.recipient != recipient
                || peer.original_deadline != self.ticket.original_deadline
            {
                return Err(CoordinatorError::IdentityMismatch(
                    "actual peer source ticket/view",
                ));
            }
        }
        Ok(AcceptedRoleOrigin {
            frame,
            ticket: self.ticket,
            output_digest: self.output_digest,
            kind: peer.map_or(
                AcceptedRoleOriginKind::LocalAccepted,
                AcceptedRoleOriginKind::PeerQuestion,
            ),
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
// A bounded value seal preserves the public Copy origin contract. Heap-backed
// ownership would change that contract; this is not a measured memory claim.
#[allow(clippy::large_enum_variant)]
pub enum AcceptedRoleOriginKind {
    LocalAccepted,
    PeerQuestion(rz_contracts::rovent::QuestionEnvelope),
}

/// Opaque origin relation, not a peer question and not new execution credit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AcceptedRoleOrigin {
    frame: super::FrameId,
    ticket: RoleTicketEnvelope,
    output_digest: Digest,
    kind: AcceptedRoleOriginKind,
}
impl AcceptedRoleOrigin {
    pub fn frame(&self) -> super::FrameId {
        self.frame
    }
    pub fn ticket(&self) -> &RoleTicketEnvelope {
        &self.ticket
    }
    pub fn output_digest(&self) -> Digest {
        self.output_digest
    }
    pub fn kind(&self) -> AcceptedRoleOriginKind {
        self.kind
    }
    pub fn matches(&self, accepted: &AcceptedRoleStep) -> bool {
        self.ticket == accepted.ticket && self.output_digest == accepted.output_digest
    }
    /// Invocation seal includes actual source ownership/issuance. It is kept
    /// outside the model input and semantic deduplication digest.
    pub fn seal_digest(&self) -> Digest {
        let mut hash = Sha256::new();
        hash.update(b"rz-rovent-accepted-local-origin-seal/1\0");
        hash.update(self.frame.sequence.to_le_bytes());
        hash.update(self.ticket.id.sequence.to_le_bytes());
        hash.update(self.ticket.authority.epoch.0.to_le_bytes());
        hash.update(self.ticket.authority.game.0.to_le_bytes());
        hash.update(self.ticket.authority.root.0.to_le_bytes());
        hash.update(self.ticket.authority.implementation.0);
        hash.update(self.ticket.question.0);
        hash.update(self.ticket.view.view_digest.0);
        hash.update(self.ticket.view.view_sequence.to_le_bytes());
        hash.update(self.output_digest.0);
        for digest in [
            self.ticket.representation.input,
            self.ticket.representation.history,
            self.ticket.representation.candidates,
            self.ticket.representation.mask_positions,
            self.ticket.representation.model,
            self.ticket.representation.encoding,
        ] {
            hash.update(digest.0);
        }
        hash.update(self.ticket.representation.frozen_epoch.to_le_bytes());
        hash.update(self.ticket.representation.record_revision.to_le_bytes());
        hash.update([self.ticket.role as u8]);
        match self.ticket.representation.precision {
            rz_contracts::PrecisionProfile::Fp32 => hash.update([0]),
            rz_contracts::PrecisionProfile::Fp16 => hash.update([1]),
            rz_contracts::PrecisionProfile::Bf16 => hash.update([2]),
            rz_contracts::PrecisionProfile::Quantized(profile) => {
                hash.update([3]);
                hash.update(profile.0);
            }
        }
        hash.update(self.ticket.original_deadline.at.0.to_le_bytes());
        match self.kind {
            AcceptedRoleOriginKind::LocalAccepted => hash.update([0]),
            AcceptedRoleOriginKind::PeerQuestion(peer) => {
                hash.update([1]);
                hash.update(peer.id.sequence.to_le_bytes());
                hash.update(peer.question_digest.0);
            }
        }
        Digest(hash.finalize().into())
    }
}

impl<V, O> RoleTicketRegistry<V, O> {
    /// Register only the verifier emitted by the actual spawned service owner.
    /// It cannot be replaced later to make another owner's receipt succeed.
    pub fn register_native_acceptance_verifier(
        &mut self,
        verifier: NativeAcceptanceVerifier,
    ) -> Result<(), CoordinatorError> {
        if !self.admission_open() {
            return Err(CoordinatorError::AdmissionClosed);
        }
        if self.native_acceptance_verifier.is_some() {
            return Err(CoordinatorError::Duplicate);
        }
        self.native_acceptance_verifier = Some(verifier);
        Ok(())
    }
    pub fn issue_native_acceptance(
        &mut self,
        ticket: &RoleTicketEnvelope,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<NativeAcceptancePermit, CoordinatorError> {
        ticket.validate(clock, now, self.token_budget)?;
        if self.native_acceptance_verifier.is_none() {
            return Err(CoordinatorError::MissingEvidence);
        }
        let held = self.match_ticket_mut(ticket)?;
        if held.phase != TicketPhase::Delivered || held.revoked || !held.successful_payload {
            return Err(CoordinatorError::InvalidTransition(
                "native acceptance requires original eligible delivery",
            ));
        }
        if held.native_acceptance_issued.is_some() || held.native_acceptance_confirmed {
            return Err(CoordinatorError::Duplicate);
        }
        let output_digest = match ticket.role {
            Role::Proposer => held
                .proposer_ranking
                .as_ref()
                .map(|head| head.output_digest),
            Role::Validator => held
                .validator_ranking
                .as_ref()
                .map(|head| head.output_digest),
            Role::Critic => None,
        }
        .ok_or(CoordinatorError::MissingEvidence)?;
        let seal = NativeAcceptanceSeal {
            ticket: *ticket,
            output_digest,
            reciprocal: false,
        };
        held.native_acceptance_issued = Some(seal);
        Ok(NativeAcceptancePermit { seal })
    }
    #[expect(
        clippy::result_large_err,
        reason = "return the original linear owner receipt intact on rejection without allocating during failure or memory pressure"
    )]
    pub fn confirm_native_acceptance(
        &mut self,
        receipt: NativeAcceptanceReceipt,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<(), RejectedNativeAcceptance> {
        let check = (|| {
            let verifier = self
                .native_acceptance_verifier
                .as_ref()
                .ok_or(CoordinatorError::MissingEvidence)?;
            if !Arc::ptr_eq(&verifier.identity, &receipt.identity) {
                return Err(CoordinatorError::IdentityMismatch(
                    "original native acceptance owner",
                ));
            }
            let ticket = receipt.ticket();
            ticket.validate(clock, now, self.token_budget)?;
            let held = self.match_ticket_mut(ticket)?;
            if held.phase != TicketPhase::Delivered
                || held.revoked
                || !held.successful_payload
                || held.native_acceptance_issued != Some(receipt.permit.seal)
                || held.native_acceptance_confirmed
                || receipt.permit.seal.reciprocal
            {
                return Err(CoordinatorError::IdentityMismatch(
                    "original native acceptance permit/head",
                ));
            }
            held.native_acceptance_confirmed = true;
            Ok(())
        })();
        check.map_err(|error| RejectedNativeAcceptance { error, receipt })
    }
}

impl<V, O: super::ProposerRankingOutput> RoleTicketRegistry<V, O> {
    /// Capture the actual physically completed P/C output before take_ready
    /// transfers it. Physical-unknown payloads retain the old quarantine path.
    #[expect(
        clippy::result_large_err,
        reason = "preserve original physical completion custody on rejection"
    )]
    pub fn complete_reciprocal_role(
        &mut self,
        completion: super::RoleCompletion<O>,
    ) -> Result<(), super::RejectedRoleCompletion<O>> {
        if completion.physical == super::PhysicalRoleCompletion::Unknown {
            return self.complete(completion);
        }
        let digest = match &completion.result {
            Ok(output) => {
                let output = output.proposer_ranking_output();
                if output.id != completion.ticket.id
                    || output.authority != completion.ticket.authority
                    || output.key != completion.ticket.representation
                    || !matches!(completion.ticket.role, Role::Proposer | Role::Critic)
                {
                    return Err(super::RejectedRoleCompletion {
                        error: CoordinatorError::IdentityMismatch("reciprocal original P/C output"),
                        completion,
                    });
                }
                match native_acceptance_output_digest(output) {
                    Ok(digest) => Some(digest),
                    Err(error) => return Err(super::RejectedRoleCompletion { error, completion }),
                }
            }
            Err(_) => None,
        };
        let ticket = completion.ticket;
        self.complete(completion)?;
        self.match_ticket_mut(&ticket)
            .expect("completed original ticket")
            .reciprocal_output_digest = digest;
        Ok(())
    }
}

impl<V, O> RoleTicketRegistry<V, O> {
    /// Check custody of an original reciprocal acceptance permit after this
    /// registry revoked its delivered ticket. This grants neither acceptance
    /// nor release: the caller must retain the permit/output until the actual
    /// native fence, and must independently verify its own cancellation ACK.
    pub fn validate_revoked_reciprocal_native_permit(
        &mut self,
        ticket: &RoleTicketEnvelope,
        permit: &NativeAcceptancePermit,
    ) -> Result<(), CoordinatorError> {
        let held = self.match_ticket_mut(ticket)?;
        if permit.ticket() != ticket
            || held.phase != TicketPhase::Delivered
            || !held.revoked
            || !held.successful_payload
            || held.native_acceptance_issued != Some(permit.seal)
            || held.native_acceptance_confirmed
            || !permit.seal.reciprocal
            || held.reciprocal_output_digest != Some(permit.output_digest())
        {
            return Err(CoordinatorError::IdentityMismatch(
                "revoked reciprocal original ticket/head/permit",
            ));
        }
        Ok(())
    }

    /// Match only an original issued P-ranking/V-task head permit after
    /// revocation. This neither confirms acceptance nor releases any owner:
    /// the caller must retain the returned permit/output until its actual
    /// native fence and independently validate the original cancellation ACK.
    pub fn validate_revoked_published_head_native_permit(
        &mut self,
        ticket: &RoleTicketEnvelope,
        permit: &NativeAcceptancePermit,
    ) -> Result<(), CoordinatorError> {
        let held = self.match_ticket_mut(ticket)?;
        let original_head = match ticket.role {
            Role::Proposer => held
                .proposer_ranking
                .as_ref()
                .map(|head| head.output_digest),
            Role::Validator => held
                .validator_ranking
                .as_ref()
                .map(|head| head.output_digest),
            Role::Critic => None,
        };
        if permit.ticket() != ticket
            || held.phase != TicketPhase::Delivered
            || !held.revoked
            || !held.successful_payload
            || held.native_acceptance_issued != Some(permit.seal)
            || held.native_acceptance_confirmed
            || permit.seal.reciprocal
            || original_head != Some(permit.output_digest())
        {
            return Err(CoordinatorError::IdentityMismatch(
                "revoked published original ticket/head/permit",
            ));
        }
        Ok(())
    }

    pub fn issue_reciprocal_native_acceptance(
        &mut self,
        ticket: &RoleTicketEnvelope,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<NativeAcceptancePermit, CoordinatorError> {
        ticket.validate(clock, now, self.token_budget)?;
        if self.native_acceptance_verifier.is_none() {
            return Err(CoordinatorError::MissingEvidence);
        }
        let held = self.match_ticket_mut(ticket)?;
        if held.phase != TicketPhase::Delivered || held.revoked || !held.successful_payload {
            return Err(CoordinatorError::InvalidTransition(
                "reciprocal acceptance requires original eligible delivery",
            ));
        }
        if held.native_acceptance_issued.is_some() || held.native_acceptance_confirmed {
            return Err(CoordinatorError::Duplicate);
        }
        let seal = NativeAcceptanceSeal {
            ticket: *ticket,
            output_digest: held
                .reciprocal_output_digest
                .ok_or(CoordinatorError::MissingEvidence)?,
            reciprocal: true,
        };
        held.native_acceptance_issued = Some(seal);
        Ok(NativeAcceptancePermit { seal })
    }

    #[expect(
        clippy::result_large_err,
        reason = "retain the original owner success receipt on rejection"
    )]
    pub fn accept_reciprocal_native(
        &mut self,
        receipt: NativeAcceptanceReceipt,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<AcceptedRoleStep, RejectedNativeAcceptance> {
        let checked = (|| {
            let verifier = self
                .native_acceptance_verifier
                .as_ref()
                .ok_or(CoordinatorError::MissingEvidence)?;
            if !Arc::ptr_eq(&verifier.identity, &receipt.identity) {
                return Err(CoordinatorError::IdentityMismatch(
                    "original reciprocal native owner",
                ));
            }
            let ticket = receipt.ticket();
            ticket.validate(clock, now, self.token_budget)?;
            let held = self.match_ticket_mut(ticket)?;
            if held.phase != TicketPhase::Delivered
                || held.revoked
                || !held.successful_payload
                || held.native_acceptance_issued != Some(receipt.permit.seal)
                || held.native_acceptance_confirmed
                || !receipt.permit.seal.reciprocal
                || held.reciprocal_output_digest != Some(receipt.output_digest())
            {
                return Err(CoordinatorError::IdentityMismatch(
                    "reciprocal original ticket/head/permit",
                ));
            }
            Ok(())
        })();
        if let Err(error) = checked {
            return Err(RejectedNativeAcceptance { error, receipt });
        }
        let step = AcceptedRoleStep {
            ticket: *receipt.ticket(),
            output_digest: receipt.output_digest(),
        };
        self.retire_accepted(step.ticket.id);
        Ok(step)
    }
}

/// Share the exact completed P/V head identity with the native service. This
/// function is an identity/numerical validator and grants no acceptance rights.
pub fn native_acceptance_output_digest(output: &RoleOutput) -> Result<Digest, CoordinatorError> {
    if output.private_latent.len() != 16 * 384
        || output.private_latent.iter().any(|v| !v.is_finite())
    {
        return Err(CoordinatorError::InvalidTransition(
            "native acceptance latent shape/numerical value",
        ));
    }
    let mut hash = Sha256::new();
    match &output.payload {
        RolePayload::Proposal {
            candidate_logits,
            wdl,
        } if output.key.role == Role::Proposer => {
            if candidate_logits.is_empty()
                || candidate_logits.len() > 256
                || candidate_logits.iter().any(|v| !v.is_finite())
            {
                return Err(CoordinatorError::InvalidTransition(
                    "native acceptance proposer head",
                ));
            }
            let [win, draw, loss] = wdl.probabilities();
            Wdl::try_new(win, draw, loss, 1e-4)?;
            hash.update(b"rz-pals-accepted-proposer-head/1\0");
            hash.update((candidate_logits.len() as u64).to_le_bytes());
            for value in candidate_logits.iter().copied().chain(wdl.probabilities()) {
                hash.update(value.to_bits().to_le_bytes());
            }
        }
        RolePayload::TaskRanking { task_logits } if output.key.role == Role::Validator => {
            if task_logits.len() != 7 || task_logits.iter().any(|v| !v.is_finite()) {
                return Err(CoordinatorError::InvalidTransition(
                    "native acceptance validator head",
                ));
            }
            hash.update(b"rz-pals-accepted-validator-head/1\0");
            for value in task_logits {
                hash.update(value.to_bits().to_le_bytes());
            }
        }
        RolePayload::Counterexample {
            candidate_logits,
            divergence_logits,
            wdl,
        } if output.key.role == Role::Critic => {
            if (candidate_logits.is_empty() && divergence_logits.is_empty())
                || candidate_logits.len() > 256
                || divergence_logits.len() > 256
                || candidate_logits
                    .iter()
                    .chain(divergence_logits)
                    .any(|v| !v.is_finite())
            {
                return Err(CoordinatorError::InvalidTransition(
                    "native acceptance critic head",
                ));
            }
            let [win, draw, loss] = wdl.probabilities();
            Wdl::try_new(win, draw, loss, 1e-4)?;
            hash.update(b"rz-rovent-accepted-critic-head/1\0");
            hash.update((candidate_logits.len() as u64).to_le_bytes());
            hash.update((divergence_logits.len() as u64).to_le_bytes());
            for value in candidate_logits
                .iter()
                .copied()
                .chain(divergence_logits.iter().copied())
                .chain(wdl.probabilities())
            {
                hash.update(value.to_bits().to_le_bytes());
            }
        }
        _ => {
            return Err(CoordinatorError::InvalidTransition(
                "native acceptance captured P/V head required",
            ));
        }
    }
    Ok(Digest(hash.finalize().into()))
}
