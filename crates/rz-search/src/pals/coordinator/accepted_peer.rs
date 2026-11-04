//! Shared accepted-source semantics for the actual PCR and P/C producers.
//!
//! This digest grants no admission. Each port still checks its private Q,
//! original Rules/first move, model namespace, budget and selected child.
//! Physical issuance belongs to AcceptedRoleOrigin::seal_digest instead.
use super::{AcceptedRoleOrigin, AcceptedRoleOriginKind, CoordinatorError};
use crate::pals::review_context::ReviewView;
use rz_contracts::Digest;
use rz_contracts::pals::{Role, RoleOutput};
use sha2::{Digest as _, Sha256};

/// Freeze while the actual accepted input/head are still held by their owner.
/// Plans retain these copied facts, not historical view pins or raw latents.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct AcceptedPeerSourceSeal {
    origin: AcceptedRoleOrigin,
    source_view: Digest,
}
impl AcceptedPeerSourceSeal {
    pub(crate) fn freeze(
        origin: AcceptedRoleOrigin,
        source_view: &ReviewView,
        output: &RoleOutput,
    ) -> Result<Self, CoordinatorError> {
        let ticket = origin.ticket();
        if !matches!(ticket.role, Role::Proposer | Role::Critic)
            || output.id != ticket.id
            || output.authority != ticket.authority
            || output.key != ticket.representation
            || super::native_acceptance_output_digest(output)? != origin.output_digest()
            || ticket.view.authority != ticket.authority
            || ticket.view.view_sequence != source_view.sequence()
            || ticket.view.view_digest.0 != source_view.seal_digest()
            || ticket.view.context_policy.0 != source_view.policy_digest()
            || ticket.view.actual_tokens != source_view.token_count()
            || usize::from(ticket.view.selected_records) != source_view.records().len()
            || usize::from(ticket.view.anchors) != source_view.anchors().len()
            || ticket.view.length_bucket != source_view.length_bucket()
            || source_view.generation() != ticket.authority.root.0
        {
            return Err(CoordinatorError::IdentityMismatch(
                "accepted peer original source head/view",
            ));
        }
        Ok(Self {
            origin,
            source_view: Digest(source_view.digest()),
        })
    }

    pub(crate) fn origin(&self) -> &AcceptedRoleOrigin {
        &self.origin
    }
    pub(crate) fn source_view_digest(&self) -> Digest {
        self.source_view
    }
}

pub(crate) fn accepted_peer_semantic_digest(
    candidate_semantics: Digest,
    source: &AcceptedPeerSourceSeal,
) -> Digest {
    let mut hash = Sha256::new();
    // Keep the existing PCR bytes. Sequence, request/frame IDs and owner
    // issuance do not turn an unchanged actual question into new progress.
    hash.update(b"rz-rovent-accepted-peer-query/1\0");
    hash.update(candidate_semantics.0);
    hash.update(source.source_view_digest().0);
    hash.update(source.origin().output_digest().0);
    hash.update([match source.origin().kind() {
        AcceptedRoleOriginKind::LocalAccepted => 0,
        AcceptedRoleOriginKind::PeerQuestion(_) => 1,
    }]);
    Digest(hash.finalize().into())
}
