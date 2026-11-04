//! Single-writer continuation and ticket authority for PALS coordination.
//!
//! This module never performs a model call, CPU search, archive I/O or pipe
//! drain. Workers execute the emitted work; the manager admits their original
//! sealed responses and commits validated Rules/store evidence. The historical
//! synchronous `RoleModel` remains available alongside this facade.

mod accepted_peer;
mod external_binding;
mod machine;
mod native_acceptance;
mod profiled_external_binding;
mod ranking;
#[cfg(test)]
mod test_support;
mod ticket_observation;
mod tickets;
mod validator;

pub(crate) use accepted_peer::{AcceptedPeerSourceSeal, accepted_peer_semantic_digest};
pub use external_binding::*;
pub use machine::*;
pub use native_acceptance::*;
pub use profiled_external_binding::*;
pub use ranking::*;
pub use ticket_observation::{RoleTicketCumulativeObservation, RoleTicketPhaseOccupancy};
pub use tickets::*;
pub use validator::*;

use rz_contracts::ContractError;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CoordinatorError {
    Contract(ContractError),
    AdmissionClosed,
    Capacity(&'static str),
    IdentityMismatch(&'static str),
    InvalidTransition(&'static str),
    /// This exact input/evidence was already retired. Optional follow-ups may
    /// skip it, without minting a frame, dependency or additional work credit.
    ReciprocalNoProgress,
    Duplicate,
    UnknownRequest,
    Deadline,
    SequenceExhausted,
    MissingEvidence,
}
impl From<ContractError> for CoordinatorError {
    fn from(value: ContractError) -> Self {
        Self::Contract(value)
    }
}
impl std::fmt::Display for CoordinatorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PALS coordinator: {self:?}")
    }
}
impl std::error::Error for CoordinatorError {}

/// Compiled implementation provenance only. This does not revise the model's
/// input semantics or the common evaluation/PALS contract revision.
pub fn coordinator_source_digest() -> rz_contracts::Digest {
    use sha2::{Digest as _, Sha256};
    let mut hash = Sha256::new();
    hash.update(b"rz-pals-adaptive-coordinator-source/1\0");
    for (name, source) in [
        (
            "coordinator.rs",
            include_bytes!("coordinator.rs").as_slice(),
        ),
        (
            "coordinator/machine.rs",
            include_bytes!("coordinator/machine.rs").as_slice(),
        ),
        (
            "coordinator/reciprocal.rs",
            include_bytes!("coordinator/reciprocal.rs").as_slice(),
        ),
        (
            "coordinator/reciprocal/utility_dependency.rs",
            include_bytes!("coordinator/reciprocal/utility_dependency.rs").as_slice(),
        ),
        (
            "coordinator/tickets.rs",
            include_bytes!("coordinator/tickets.rs").as_slice(),
        ),
        (
            "coordinator/ticket_observation.rs",
            include_bytes!("coordinator/ticket_observation.rs").as_slice(),
        ),
        (
            "coordinator/ranking.rs",
            include_bytes!("coordinator/ranking.rs").as_slice(),
        ),
        (
            "coordinator/validator.rs",
            include_bytes!("coordinator/validator.rs").as_slice(),
        ),
        (
            "coordinator/native_acceptance.rs",
            include_bytes!("coordinator/native_acceptance.rs").as_slice(),
        ),
        (
            "coordinator/accepted_peer.rs",
            include_bytes!("coordinator/accepted_peer.rs").as_slice(),
        ),
        (
            "coordinator/external_binding.rs",
            include_bytes!("coordinator/external_binding.rs").as_slice(),
        ),
        (
            "coordinator/profiled_external_binding.rs",
            include_bytes!("coordinator/profiled_external_binding.rs").as_slice(),
        ),
    ] {
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name.as_bytes());
        hash.update((source.len() as u64).to_le_bytes());
        hash.update(source);
    }
    rz_contracts::Digest(hash.finalize().into())
}
